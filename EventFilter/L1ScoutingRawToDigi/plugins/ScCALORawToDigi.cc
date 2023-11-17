#include "EventFilter/L1ScoutingRawToDigi/plugins/ScCALORawToDigi.h"

ScCaloRawToDigi::ScCaloRawToDigi(const edm::ParameterSet& iConfig) {
  using namespace edm;
  using namespace scoutingRun3;
  srcInputTag  = iConfig.getParameter<InputTag>( "srcInputTag" );
  debug = iConfig.getUntrackedParameter<bool>("debug", false);

  // produces<JetOrbitCollection>().setBranchAlias( "JetOrbitCollection" );
  // produces<TauOrbitCollection>().setBranchAlias( "TauOrbitCollection" );
  // produces<EGammaOrbitCollection>().setBranchAlias( "EGammaOrbitCollection" );
  // produces<EtSumOrbitCollection>().setBranchAlias( "EtSumOrbitCollection" );

  produces<ScJetOrbitCollection>().setBranchAlias( "ScJetOrbitCollection" );
  produces<ScTauOrbitCollection>().setBranchAlias( "ScTauOrbitCollection" );
  produces<ScEGammaOrbitCollection>().setBranchAlias( "ScEGammaOrbitCollection" );
  produces<ScEtSumOrbitCollection>().setBranchAlias( "ScEtSumOrbitCollection" );
  
  rawToken = consumes<SRDCollection>(srcInputTag);
  }

ScCaloRawToDigi::~ScCaloRawToDigi() {};

void ScCaloRawToDigi::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace scoutingRun3;

  Handle<SRDCollection> ScoutingRawDataCollection;
  iEvent.getByToken( rawToken, ScoutingRawDataCollection );

  const FEDRawData& sourceRawData = ScoutingRawDataCollection->FEDData(SDSNumbering::CaloSDSID);
  size_t orbitSize = sourceRawData.size();

  // std::unique_ptr<JetOrbitCollection> unpackedJets(new JetOrbitCollection);
  // std::unique_ptr<TauOrbitCollection> unpackedTaus(new TauOrbitCollection);
  // std::unique_ptr<EGammaOrbitCollection> unpackedEGammas(new EGammaOrbitCollection);
  // std::unique_ptr<EtSumOrbitCollection> unpackedEtSums(new EtSumOrbitCollection);

  std::unique_ptr<ScJetOrbitCollection> unpackedJets(new ScJetOrbitCollection);
  std::unique_ptr<ScTauOrbitCollection> unpackedTaus(new ScTauOrbitCollection);
  std::unique_ptr<ScEGammaOrbitCollection> unpackedEGammas(new ScEGammaOrbitCollection);
  std::unique_ptr<ScEtSumOrbitCollection> unpackedEtSums(new ScEtSumOrbitCollection);
  
  if((sourceRawData.size()==0) && debug ){
    std::cout << "No raw data for CALO FED\n";  
  }

  unpackOrbit(
    unpackedJets.get(), unpackedTaus.get(),
    unpackedEGammas.get(), unpackedEtSums.get(),
    sourceRawData.data(), orbitSize
  ); 

  // store collections in the event
  iEvent.put( std::move(unpackedJets) );
  iEvent.put( std::move(unpackedTaus) );
  iEvent.put( std::move(unpackedEGammas) );
  iEvent.put( std::move(unpackedEtSums) );
}

void ScCaloRawToDigi::unpackOrbit(
  scoutingRun3::ScJetOrbitCollection* jets, scoutingRun3::ScTauOrbitCollection* taus,
  scoutingRun3::ScEGammaOrbitCollection* eGammas, scoutingRun3::ScEtSumOrbitCollection* etSums,
  const unsigned char* buf, size_t len
  ){
  
  using namespace scoutingRun3;
  
  size_t pos = 0;

  while (pos < len) {
    
    assert(pos+ (4+4+4+56*4) <= len); //sizeof(demux::block)

    demux::block *bl = (demux::block *)(buf + pos);
    pos += 4+4+4+56*4;

    assert(pos <= len);
    uint32_t orbit = bl->orbit & 0x7FFFFFFF;
    uint32_t bx = bl->bx;

    if(debug) {
      std::cout << " CALO Orbit " << orbit << ", BX -> "<< bx << std::endl;
    }

    math::PtEtaPhiMLorentzVector vec;

    int32_t ET(0), Eta(0), Phi(0), Iso(0);
    //float   fET(0), fEta(0), fPhi(0);

    // unpack jets from first link
    for (uint32_t i=0; i<6; i++) {
      ET = ((bl->jet1[i] >> demux::shiftsJet::ET)  & demux::masksJet::ET);
      
      if (ET != 0) {
        Eta = ((bl->jet1[i] >> demux::shiftsJet::eta) & demux::masksJet::eta);
        Phi = ((bl->jet1[i] >> demux::shiftsJet::phi) & demux::masksJet::phi);
        Iso = 0;

        if (Eta > 127) Eta = Eta - 256;
        // fPhi = Phi * demux::scales::phi_scale; fPhi = fPhi>=2.*M_PI ? fPhi-2.*M_PI : fPhi;
        // fEta = Eta * demux::scales::eta_scale;
        // fET  = ET  * demux::scales::et_scale;
        
        //l1t::Jet jet(math::PtEtaPhiMLorentzVector(fET, fEta, fPhi, 0.), ET, Eta, Phi, Iso);
        ScJet jet1(ET, Eta, Phi, 0);
        jets->addBxObject(bx, jet1);


        if (debug){
          std::cout << "--- Jet link 1 ---\n";
          std::cout <<"\tEt  [GeV/Hw]: "<< jet1.getEt() << "/" << jet1.getHwEt() << "\n";
          std::cout <<"\tEta [rad/Hw]: " << jet1.getEta() << "/" << jet1.getHwEta() << "\n";
          std::cout <<"\tPhi [rad/Hw]: " << jet1.getPhi() << "/" << jet1.getHwPhi() << "\n";
        }
      } 
    } // end link1 jet unpacking loop
    
    // unpack jets from second link
    for (uint32_t i=0; i<6; i++) {
      ET = ((bl->jet2[i] >> demux::shiftsJet::ET)  & demux::masksJet::ET);
      
      if (ET != 0) {
        Eta = ((bl->jet2[i] >> demux::shiftsJet::eta) & demux::masksJet::eta);
        Phi = ((bl->jet2[i] >> demux::shiftsJet::phi) & demux::masksJet::phi);
        Iso = 0;

        if (Eta > 127) Eta = Eta - 256;
        // fPhi = Phi * demux::scales::phi_scale; fPhi = fPhi>=2.*M_PI ? fPhi-2.*M_PI : fPhi;
        // fEta = Eta * demux::scales::eta_scale;
        // fET  = ET  * demux::scales::et_scale;

        //l1t::Jet jet(math::PtEtaPhiMLorentzVector(fET, fEta, fPhi, 0.), ET, Eta, Phi, Iso);
        ScJet jet2(ET, Eta, Phi, 0);
        jets->addBxObject(bx, jet2);

        if (debug){
          std::cout << "--- Jet link 2 ---\n";
          std::cout <<"\tEt  [GeV/Hw]: " << jet2.getEt() << "/" << jet2.getHwEt() << "\n";
          std::cout <<"\tEta [rad/Hw]: " << jet2.getEta() << "/" << jet2.getHwEta() << "\n";
          std::cout <<"\tPhi [rad/Hw]: " << jet2.getPhi() << "/" << jet2.getHwPhi() << "\n";
        }
      } 
    } // end link1 jet unpacking loop

    // unpack eg from first link
    for (uint32_t i=0; i<6; i++) {
      ET = ((bl->egamma1[i] >> demux::shiftsEGamma::ET)  & demux::masksEGamma::ET);
      if (ET != 0) {
        Eta   = ((bl->egamma1[i] >> demux::shiftsEGamma::eta) & demux::masksEGamma::eta);
        Phi   = ((bl->egamma1[i] >> demux::shiftsEGamma::phi) & demux::masksEGamma::phi);
        Iso   = ((bl->egamma1[i] >> demux::shiftsEGamma::iso) & demux::masksEGamma::iso);

        if (Eta > 127) Eta = Eta - 256;
        // fPhi = Phi * demux::scales::phi_scale; fPhi = fPhi>=2.*M_PI ? fPhi-2.*M_PI : fPhi;
        // fEta = Eta * demux::scales::eta_scale;
        // fET  = ET  * demux::scales::et_scale;
        
        //l1t::EGamma eGamma(math::PtEtaPhiMLorentzVector(fET, fEta, fPhi, 0.), ET, Eta, Phi, 0, Iso);
        ScEGamma eGamma1(ET, Eta, Phi, Iso);
        eGammas->addBxObject(bx, eGamma1);
        
        if (debug){
          std::cout << "--- E/g link 1 ---\n";
          std::cout <<"\tEt  [GeV/Hw]: " << eGamma1.getEt()  << "/" << eGamma1.getHwEt() << "\n";
          std::cout <<"\tEta [rad/Hw]: " << eGamma1.getEta() << "/" << eGamma1.getHwEta() << "\n";
          std::cout <<"\tPhi [rad/Hw]: " << eGamma1.getPhi() << "/" << eGamma1.getHwPhi() << "\n";
          std::cout <<"\tIso [Hw]: " << eGamma1.getIso() << "\n";
        }
      }
    } // end eg link 1

    // unpack eg from second link link
    for (uint32_t i=0; i<6; i++) {
      ET = ((bl->egamma2[i] >> demux::shiftsEGamma::ET)  & demux::masksEGamma::ET);
      if (ET != 0) {
        Eta   = ((bl->egamma2[i] >> demux::shiftsEGamma::eta) & demux::masksEGamma::eta);
        Phi   = ((bl->egamma2[i] >> demux::shiftsEGamma::phi) & demux::masksEGamma::phi);
        Iso   = ((bl->egamma2[i] >> demux::shiftsEGamma::iso) & demux::masksEGamma::iso);

        if (Eta > 127) Eta = Eta - 256;
        // fPhi = Phi * demux::scales::phi_scale; fPhi = fPhi>=2.*M_PI ? fPhi-2.*M_PI : fPhi;
        // fEta = Eta * demux::scales::eta_scale;
        // fET  = ET  * demux::scales::et_scale;

        //l1t::EGamma eGamma(math::PtEtaPhiMLorentzVector(fET, fEta, fPhi, 0.), ET, Eta, Phi, 0, Iso);
        ScEGamma eGamma2(ET, Eta, Phi, Iso);
        eGammas->addBxObject(bx, eGamma2);

        if (debug){
          std::cout << "--- E/g link 2 ---\n";
          std::cout <<"\tEt  [GeV/Hw]: " << eGamma2.getEt()  << "/" << eGamma2.getHwEt() << "\n";
          std::cout <<"\tEta [rad/Hw]: " << eGamma2.getEta() << "/" << eGamma2.getHwEta() << "\n";
          std::cout <<"\tPhi [rad/Hw]: " << eGamma2.getPhi() << "/" << eGamma2.getHwPhi() << "\n";
          std::cout <<"\tIso [Hw]: " << eGamma2.getIso() << "\n";
        }
      }

    } // end of eg unpacker

    // unpack taus from first link
    for (uint32_t i=0; i<6; i++) { 
      ET = ((bl->tau1[i] >> demux::shiftsTau::ET)  & demux::masksTau::ET);
      if (ET != 0) {
          Eta   = ((bl->tau1[i] >> demux::shiftsTau::eta) & demux::masksTau::eta);
          Phi   = ((bl->tau1[i] >> demux::shiftsTau::phi) & demux::masksTau::phi);
          Iso   = ((bl->tau1[i] >> demux::shiftsTau::iso) & demux::masksTau::iso);

          if (Eta > 127) Eta = Eta - 256;
          // fPhi = Phi * demux::scales::phi_scale; fPhi = fPhi>=2.*M_PI ? fPhi-2.*M_PI : fPhi;
          // fEta = Eta * demux::scales::eta_scale;
          // fET  = ET  * demux::scales::et_scale;

          //l1t::Tau tau(math::PtEtaPhiMLorentzVector(fET, fEta, fPhi, 0.), ET, Eta, Phi, 0, Iso);
          ScTau tau1(ET, Eta, Phi, Iso);
          taus->addBxObject(bx, tau1);

          if (debug){
            std::cout << "--- Tau link 1 ---\n";
            std::cout <<"\tEt  [GeV/Hw]: " << tau1.getEt()  << "/" << tau1.getHwEt() << "\n";
            std::cout <<"\tEta [rad/Hw]: " << tau1.getEta() << "/" << tau1.getHwEta() << "\n";
            std::cout <<"\tPhi [rad/Hw]: " << tau1.getPhi() << "/" << tau1.getHwPhi() << "\n";
            std::cout <<"\tIso [Hw]: " << tau1.getIso() << "\n";
          }
      }
    } // end tau link 1

    // unpack taus from second link
    for (uint32_t i=0; i<6; i++) { 
      ET = ((bl->tau2[i] >> demux::shiftsTau::ET)  & demux::masksTau::ET);
      if (ET != 0) {
          Eta   = ((bl->tau2[i] >> demux::shiftsTau::eta) & demux::masksTau::eta);
          Phi   = ((bl->tau2[i] >> demux::shiftsTau::phi) & demux::masksTau::phi);
          Iso   = ((bl->tau2[i] >> demux::shiftsTau::iso) & demux::masksTau::iso);

          if (Eta > 127) Eta = Eta - 256;
          // fPhi = Phi * demux::scales::phi_scale; fPhi = fPhi>=2.*M_PI ? fPhi-2.*M_PI : fPhi;
          // fEta = Eta * demux::scales::eta_scale;
          // fET  = ET  * demux::scales::et_scale;

          //l1t::Tau tau(math::PtEtaPhiMLorentzVector(fET, fEta, fPhi, 0.), ET, Eta, Phi, 0, Iso);
          ScTau tau2(ET, Eta, Phi, Iso);
          taus->addBxObject(bx, tau2);

          if (debug){
            std::cout << "--- Tau link 2 ---\n";
            std::cout <<"\tEt  [GeV/Hw]: " << tau2.getEt()  << "/" << tau2.getHwEt() << "\n";
            std::cout <<"\tEta [rad/Hw]: " << tau2.getEta() << "/" << tau2.getHwEta() << "\n";
            std::cout <<"\tPhi [rad/Hw]: " << tau2.getPhi() << "/" << tau2.getHwPhi() << "\n";
            std::cout <<"\tIso [Hw]: " << tau2.getIso() << "\n";
          }
      }
    } // end tau unpacker

    // unpack et sums
  
    int32_t ETEt(0), HTEt(0), ETmissEt(0), ETmissPhi(0), HTmissEt(0), HTmissPhi(0);
    // float fETEt(0), fHTEt(0), fETmissEt(0), fHTmissEt(0), fETmissPhi(0), fHTmissPhi(0);

    // ET
    ETEt        = ((bl->sum[0] >> demux::shiftsESums::ETEt)        & demux::masksESums::ETEt);
    // fETEt       = ETEt * demux::scales::et_scale;
    
    //l1t::EtSum sum = l1t::EtSum(math::PtEtaPhiMLorentzVector(fETEt, 0, 0, 0.), l1t::EtSum::EtSumType::kTotalEt, ETEt);
    ScEtSum sumTotEt(ETEt, 0, l1t::EtSum::EtSumType::kTotalEt);
    etSums->addBxObject(bx, sumTotEt);

    // HT
    HTEt         = ((bl->sum[1] >> demux::shiftsESums::HTEt)         & demux::masksESums::HTEt);
    // fHTEt        = HTEt * demux::scales::et_scale;

    //sum = l1t::EtSum(math::PtEtaPhiMLorentzVector(fHTEt, 0, 0, 0.), l1t::EtSum::EtSumType::kTotalHt, HTEt);
    ScEtSum sumTotHt(HTEt, 0, l1t::EtSum::EtSumType::kTotalHt);
    etSums->addBxObject(bx, sumTotHt);

    // ETMiss
    ETmissEt        = ((bl->sum[2] >> demux::shiftsESums::ETmissEt)        & demux::masksESums::ETmissEt);
    ETmissPhi       = ((bl->sum[2] >> demux::shiftsESums::ETmissPhi)       & demux::masksESums::ETmissPhi);
    // fETmissEt  = ETmissEt  * demux::scales::et_scale;
    // fETmissPhi = ETmissPhi * demux::scales::phi_scale;
    // fETmissPhi = fETmissPhi>2.*M_PI? fETmissPhi - 2.*M_PI : fETmissPhi;
    
    //sum = l1t::EtSum(math::PtEtaPhiMLorentzVector(fETmissEt, 0, fETmissPhi, 0.), l1t::EtSum::EtSumType::kMissingEt, ETmissEt, 0, ETmissPhi);
    
    ScEtSum sumMissEt(ETmissEt, ETmissPhi, l1t::EtSum::EtSumType::kMissingEt);
    etSums->addBxObject(bx, sumMissEt);
    
    // HTMiss
    HTmissEt        = ((bl->sum[3] >> demux::shiftsESums::HTmissEt)        & demux::masksESums::HTmissEt);
    HTmissPhi       = ((bl->sum[3] >> demux::shiftsESums::HTmissPhi)       & demux::masksESums::HTmissPhi);
    // fHTmissEt  = HTmissEt  * demux::scales::et_scale;
    // fHTmissPhi = HTmissPhi * demux::scales::phi_scale;
    // fHTmissPhi = fHTmissPhi>2.*M_PI? fHTmissPhi - 2.*M_PI : fHTmissPhi;

    ScEtSum sumMissHt(HTmissEt, HTmissPhi, l1t::EtSum::EtSumType::kMissingHt);
    etSums->addBxObject(bx, sumMissHt);


    // // ETHFMiss
    // ETHFmissEt       = ((bl->sum[4] >> demux::shiftsESums::ETHFmissEt)       & demux::masksESums::ETHFmissEt);
    // ETHFmissPhi      = ((bl->sum[4] >> demux::shiftsESums::ETHFmissPhi)      & demux::masksESums::ETHFmissPhi);
    // ETHFmissASYMETHF = ((bl->sum[4] >> demux::shiftsESums::ETHFmissASYMETHF) & demux::masksESums::ETHFmissASYMETHF);
    // //ETHFmissCENT     = ((bl.sum[4] >> demux::shiftsESums::ETHFmissCENT)     & demux::masksESums::ETHFmissCENT);
    
    // sum = l1t::EtSum(*dummyLVec_, l1t::EtSum::EtSumType::kMissingEtHF, ETHFmissEt, 0, ETHFmissPhi);
    // etSums->push_back(bx, sum);
    // sum =  l1t::EtSum(*dummyLVec_, l1t::EtSum::EtSumType::kAsymEtHF, ETHFmissASYMETHF);
    // etSums->push_back(bx, sum);

   
    // // HTHFMiss
    // HTHFmissEt       = ((bl->sum[5] >> demux::shiftsESums::HTHFmissEt)       & demux::masksESums::HTHFmissEt);
    // HTHFmissPhi      = ((bl->sum[5] >> demux::shiftsESums::HTHFmissPhi)      & demux::masksESums::HTHFmissPhi);
    // HTHFmissASYMHTHF = ((bl->sum[5] >> demux::shiftsESums::HTHFmissASYMHTHF) & demux::masksESums::HTHFmissASYMHTHF);
    // //HTHFmissCENT     = ((bl->sum[5] >> demux::shiftsESums::HTHFmissCENT)     & demux::masksESums::HTHFmissCENT);

    // sum = l1t::EtSum(*dummyLVec_, l1t::EtSum::EtSumType::kMissingHtHF, HTHFmissEt, 0, HTHFmissPhi);
    // etSums->push_back(bx, sum);
    // sum =  l1t::EtSum(*dummyLVec_, l1t::EtSum::EtSumType::kAsymHtHF, HTHFmissASYMHTHF);
    // etSums->push_back(bx, sum);

    // add sums to event
    // etSums->push_back(bx, bx_etSums);

  } // end of orbit loop

  jets->flatten();
  eGammas->flatten();
  taus->flatten();
  etSums->flatten();
  
}

void ScCaloRawToDigi::unpackRawJet(std::vector<l1t::Jet>& jets, uint32_t *rawData){
  
 return; 
}

void ScCaloRawToDigi::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(ScCaloRawToDigi);
