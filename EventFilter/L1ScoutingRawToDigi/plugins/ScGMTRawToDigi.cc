#include "EventFilter/L1ScoutingRawToDigi/plugins/ScGMTRawToDigi.h"

ScGMTRawToDigi::ScGMTRawToDigi(const edm::ParameterSet& iConfig) {
  using namespace edm;
  srcInputTag  = iConfig.getParameter<InputTag>( "srcInputTag" );
  debug_ = iConfig.getUntrackedParameter<bool>("debug", false);

  //produces<scoutingRun3::MuonOrbitCollection>().setBranchAlias( "MuonOrbitCollection" );
  produces<scoutingRun3::ScMuonOrbitCollection>().setBranchAlias( "ScMuonOrbitCollection" );
  rawToken = consumes<SRDCollection>(srcInputTag);
  
  bx_muons.reserve(8);
  //dummyLVec_.reset( new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>>() );
}

ScGMTRawToDigi::~ScGMTRawToDigi() {};

void ScGMTRawToDigi::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  Handle<SRDCollection> ScoutingRawDataCollection;
  iEvent.getByToken( rawToken, ScoutingRawDataCollection );

  const FEDRawData& sourceRawData = ScoutingRawDataCollection->FEDData(SDSNumbering::GmtSDSID);
  size_t orbitSize = sourceRawData.size();

  //std::unique_ptr<scoutingRun3::MuonOrbitCollection> unpackedMuons(new scoutingRun3::MuonOrbitCollection);
  std::unique_ptr<scoutingRun3::ScMuonOrbitCollection> unpackedMuons(new scoutingRun3::ScMuonOrbitCollection);

  if((sourceRawData.size()==0) && debug_){
    std::cout << "No raw data for GMT FED\n";  
  }

  unpackOrbit(unpackedMuons.get(), sourceRawData.data(), orbitSize); 

  // store collection in the event
  iEvent.put( std::move(unpackedMuons) );
}

void ScGMTRawToDigi::unpackOrbit(
  //l1t::MuonBxCollection* muons, 
  scoutingRun3::ScMuonOrbitCollection* muons,
  const unsigned char* buf, size_t len
  ){
  
  using namespace scoutingRun3;
  size_t pos = 0;

  //muons->setBXRange(0,3565);
  
  while (pos < len) {
    assert(pos+4 <= len);
    
    // get BX header
    uint32_t header = *((uint32_t*)(buf + pos));
    pos += 4;
    // count mA and mB
    uint32_t mAcount = (header & header_masks::mAcount) >> header_shifts::mAcount;
    uint32_t mBcount = (header & header_masks::mBcount) >> header_shifts::mBcount;
    
    // declare block to read
    ugmt::block *bl = (ugmt::block *)(buf + pos);
    pos += 4 + 4 + (mAcount+mBcount)*12; 
    assert(pos <= len);

    uint32_t orbit = bl->orbit & 0x7FFFFFFF;  
    uint32_t bx = bl->bx;
   
    if (debug_){
      std::cout  << " GMT Orbit " << orbit << ", BX -> "<< bx << ", nMuons -> " << mAcount+mBcount << std::endl;
    }
    
    // Unpack muons for this BX
    
    bx_muons.clear();
    
    // cuts should be applied
    bool excludeIntermediate=true;
    int ptcut=0;
    unsigned int qualcut=0;

    for (unsigned int i=0; i<mAcount+mBcount; i++) {

      uint32_t interm = (bl->mu[i].extra >> ugmt::shiftsMuon::interm) & ugmt::masksMuon::interm;
      if (excludeIntermediate && (interm == 1)){
        if (debug_){
          std::cout << "Excluding intermediate muon\n";
        }
        continue;
      }

      uint32_t index    = (bl->mu[i].s >> ugmt::shiftsMuon::index)  & ugmt::masksMuon::index;
      uint32_t ietaextu = (bl->mu[i].f >> ugmt::shiftsMuon::etaext) & ugmt::masksMuon::etaextv;
      int32_t ietaext;
      if (((bl->mu[i].f >> ugmt::shiftsMuon::etaext) & ugmt::masksMuon::etaexts)!=0) {
          ietaext = ietaextu -= 256;
      } else {
          ietaext = ietaextu;
      }
      
      // extract pt and quality and apply cut if required
      int32_t iptuncon = (bl->mu[i].s >> ugmt::shiftsMuon::ptuncon) & ugmt::masksMuon::ptuncon;
      int32_t ipt      = (bl->mu[i].f >> ugmt::shiftsMuon::pt)      & ugmt::masksMuon::pt;
      if ((ipt-1) < ptcut) {
          continue;
      }
      uint32_t qual = (bl->mu[i].f >> ugmt::shiftsMuon::qual) & ugmt::masksMuon::qual;
      if (qual < qualcut) {
          continue;
      }
      
      // extract integer value for extrapolated phi
      int32_t iphiext = ((bl->mu[i].f >> ugmt::shiftsMuon::phiext) & ugmt::masksMuon::phiext);

      // extract integer value for extrapolated phi
      int32_t idxy = ((bl->mu[i].s >> ugmt::shiftsMuon::dxy) & ugmt::masksMuon::dxy);

      // extract iso bits and charge
      uint32_t iso = (bl->mu[i].s >> ugmt::shiftsMuon::iso) & ugmt::masksMuon::iso;
      int32_t chrg = 0;
      if (((bl->mu[i].s >> ugmt::shiftsMuon::chrgv) & ugmt::masksMuon::chrgv)==1)
          chrg=((bl->mu[i].s >> ugmt::shiftsMuon::chrg) & ugmt::masksMuon::chrg)==1 ? -1 : 1 ;

      // extract eta and phi at muon station
      int32_t  iphi  = (bl->mu[i].s >> ugmt::shiftsMuon::phi)      & ugmt::masksMuon::phi;
      uint32_t ieta1 = (bl->mu[i].extra >> ugmt::shiftsMuon::eta1) & ugmt::masksMuon::eta;
      uint32_t ieta2 = (bl->mu[i].extra >> ugmt::shiftsMuon::eta2) & ugmt::masksMuon::eta;


      uint32_t ieta_u;
      int32_t ieta;
      // checking if raw eta should be taken from muon 1 or muon 2
      if ( (bl->mu[i].extra & 0x1) == 0 ) {
          ieta_u = ieta1;
      } else {
          ieta_u = ieta2;
      }

      // two's complement
      if ( ieta_u > 256 ) {
          ieta = ieta_u - 512;
      } else {
          ieta = ieta_u;
      }

      // convert to physical units using scales
      float fpt      = (ipt     -1) * ugmt::scales::pt_scale;                 // -1 since bin 0 is for invalid muons
      float fptuncon = (iptuncon-1) * ugmt::scales::ptunconstrained_scale;    // -1 since bin 0 is for invalid muons
      float fphi     = iphi         * ugmt::scales::phi_scale;
      float fphiext  = iphiext      * ugmt::scales::phi_scale;
      float feta     = ieta         * ugmt::scales::eta_scale;
      float fetaext  = ietaext      * ugmt::scales::eta_scale;

      if (fphiext>M_PI) fphiext = fphiext - 2.*M_PI;
      if (fphi   >M_PI) fphi    = fphi    - 2.*M_PI;

      // l1t::Muon muon;
      // math::PtEtaPhiMLorentzVector vec{fpt, feta, fphi, 0.};

      // muon.setP4(vec);
      // muon.setHwPt(ipt);
      // muon.setHwEta(ieta);
      // muon.setHwPhi(iphi);
      // muon.setHwQual(qual);
      // muon.setHwCharge(chrg);
      // muon.setHwChargeValid(chrg != 0);
      // muon.setHwIso(iso);
      // muon.setTfMuonIndex(index);
      // muon.setHwEtaAtVtx(ietaext);
      // muon.setHwPhiAtVtx(iphiext);
      // muon.setEtaAtVtx(fetaext);
      // muon.setPhiAtVtx(fphiext);
      // muon.setHwPtUnconstrained(iptuncon);
      // muon.setPtUnconstrained(fptuncon);
      // muon.setHwDXY(idxy);

      // ScMuon muon;

      // muon.pt = ipt;
      // muon.eta = ieta;
      // muon.phi = iphi;
      // muon.qual = qual;
      // muon.chrg = chrg;
      // muon.chrgv = chrg!=0;
      // muon.iso = iso;
      // muon.index = index;
      // muon.etae = ietaext;
      // muon.phie = iphiext;
      // muon.ptUncon = iptuncon;

      scoutingRun3::ScMuon muon(
        ipt,
        ieta,
        iphi,
        qual,
        chrg,
        chrg!=0,
        iso,
        index,
        ietaext,
        iphiext,
        iptuncon,
        idxy
      );

      muons->addBxObject(bx, muon);

      // if (debug_){
      //   std::cout<<"--- Muon ---\n";
      //   std::cout<<"\tPt  [GeV/Hw]: " << muon.pt()  << "/" << muon.hwPt() << "\n";
      //   std::cout<<"\tEta [rad/Hw]: " << muon.eta() << "/" << muon.hwEta() << "\n";
      //   std::cout<<"\tPhi [rad/Hw]: " << muon.phi() << "/" << muon.hwPhi() << "\n";
      //   std::cout<<"\tCharge/valid: " << muon.hwCharge() << "/" << muon.hwChargeValid() << "\n";
      //   std::cout<<"\tPhiVtx  [rad/Hw]: " << muon.phiAtVtx() << "/" << muon.hwPhiAtVtx() << "\n";
      //   std::cout<<"\tEtaVtx  [rad/Hw]: " << muon.etaAtVtx() << "/" << muon.hwEtaAtVtx() << "\n";
      //   std::cout<<"\tPt uncon[GeV/Hw]: " << muon.ptUnconstrained()  << "/" << muon.hwPtUnconstrained() << "\n";
      //   std::cout<<"\tDxy: " << muon.hwDXY() << "\n";
      //   std::cout<<"\tTF index: " << muon.tfMuonIndex() << "\n";
      // }

    } // end of bx

  } // end orbit while loop

  muons->flatten();

}

void ScGMTRawToDigi::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(ScGMTRawToDigi);
