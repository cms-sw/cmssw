#include "EventFilter/L1ScoutingRawToDigi/plugins/ScCALORawToDigi.h"

ScCaloRawToDigi::ScCaloRawToDigi(const edm::ParameterSet& iConfig) {
  using namespace edm;
  using namespace scoutingRun3;
  srcInputTag  = iConfig.getParameter<InputTag>("srcInputTag");
  debug_ = iConfig.getUntrackedParameter<bool>("debug", false);

  orbitBufferJets_    = std::vector<std::vector<scoutingRun3::ScJet>>(3565);
  orbitBufferEGammas_ = std::vector<std::vector<scoutingRun3::ScEGamma>>(3565);
  orbitBufferTaus_    = std::vector<std::vector<scoutingRun3::ScTau>>(3565);
  orbitBufferEtSums_  = std::vector<std::vector<scoutingRun3::ScEtSum>>(3565);

  nJetsOrbit_=0; nEGammasOrbit_=0; nTausOrbit_=0; nEtSumsOrbit_=0;

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

  std::unique_ptr<ScJetOrbitCollection> unpackedJets(new ScJetOrbitCollection);
  std::unique_ptr<ScTauOrbitCollection> unpackedTaus(new ScTauOrbitCollection);
  std::unique_ptr<ScEGammaOrbitCollection> unpackedEGammas(new ScEGammaOrbitCollection);
  std::unique_ptr<ScEtSumOrbitCollection> unpackedEtSums(new ScEtSumOrbitCollection);
  
  if((sourceRawData.size()==0) && debug_ ){
    std::cout << "No raw data for CALO source\n";  
  }

  // unpack current orbit and store data into the orbitBufferr
  unpackOrbit(sourceRawData.data(), orbitSize);

  // fill orbit collection and clear the Bx buffer vector
  unpackedJets->fillAndClear(orbitBufferJets_, nJetsOrbit_);
  unpackedEGammas->fillAndClear(orbitBufferEGammas_, nEGammasOrbit_);
  unpackedTaus->fillAndClear(orbitBufferTaus_, nTausOrbit_);
  unpackedEtSums->fillAndClear(orbitBufferEtSums_, nEtSumsOrbit_);

  // store collections in the event
  iEvent.put( std::move(unpackedJets) );
  iEvent.put( std::move(unpackedTaus) );
  iEvent.put( std::move(unpackedEGammas) );
  iEvent.put( std::move(unpackedEtSums) );
}

void ScCaloRawToDigi::unpackOrbit(
    const unsigned char* buf, size_t len
  ){
  using namespace scoutingRun3;

  // reset counters
  nJetsOrbit_=0; nEGammasOrbit_=0; nTausOrbit_=0; nEtSumsOrbit_=0;
  
  size_t pos = 0;

  while (pos < len) {
    
    assert( pos+sizeof(demux::block) <= len );

    demux::block *bl = (demux::block *)(buf + pos);
    pos += sizeof(demux::block);

    assert(pos <= len);
    uint32_t orbit = bl->orbit & 0x7FFFFFFF;
    uint32_t bx = bl->bx;

    if(debug_) {
      std::cout << "CALO Orbit " << orbit << ", BX -> "<< bx << std::endl;
    }

    // unpack jets from first link
    if (debug_) std::cout << "--- Jets link 1 ---\n";
    unpackLinkJets(bl->jet1, bx);
    
    // unpack jets from second link
    if (debug_) std::cout << "--- Jets link 2 ---\n";
    unpackLinkJets(bl->jet2, bx);

    // unpack eg from first link
    if (debug_) std::cout << "--- E/g link 1 ---\n";
    unpackLinkEGammas(bl->egamma1, bx);
  
    // unpack eg from second link link
    if (debug_) std::cout << "--- E/g link 2 ---\n";
    unpackLinkEGammas(bl->egamma2, bx);

    // unpack taus from first link
    if (debug_) std::cout << "--- Taus link 1 ---\n";
    unpackLinkTaus(bl->tau1, bx);
    
    // unpack taus from second link
    if (debug_) std::cout << "--- Taus link 2 ---\n";
    unpackLinkTaus(bl->tau2, bx);

    // unpack et sums
    if (debug_) std::cout << "--- Sums ---\n";
    unpackEtSums(bl->sum, bx);

  } // end of orbit loop
  
}
 
void ScCaloRawToDigi::unpackLinkJets(uint32_t* dataBlock, int bx){
  using namespace scoutingRun3;

  int32_t ET(0), Eta(0), Phi(0);
  for (uint32_t i=0; i<6; i++) {
    ET = ((dataBlock[i] >> demux::shiftsJet::ET)  & demux::masksJet::ET);
    
    if (ET != 0) {
      Eta = ((dataBlock[i] >> demux::shiftsJet::eta) & demux::masksJet::eta);
      Phi = ((dataBlock[i] >> demux::shiftsJet::phi) & demux::masksJet::phi);

      if (Eta > 127) Eta = Eta - 256;

      ScJet jet(ET, Eta, Phi, 0);
      orbitBufferJets_[bx].push_back(jet);
      nJetsOrbit_ ++;

      if (debug_){
        std::cout << "Jet " << i << std::endl;
        std::cout << "  Raw: 0x" << std::hex << dataBlock[i] << std::dec << std::endl;
        std::cout << "  Et  [GeV/Hw]: " << demux::fEt(jet.hwEt())   << "/" << jet.hwEt()  << "\n";
        std::cout << "  Eta [rad/Hw]: " << demux::fEta(jet.hwEta()) << "/" << jet.hwEta() << "\n";
        std::cout << "  Phi [rad/Hw]: " << demux::fPhi(jet.hwPhi()) << "/" << jet.hwPhi() << "\n";
      }
    } 
  } // end link jets unpacking loop 
}

void ScCaloRawToDigi::unpackLinkEGammas(uint32_t* dataBlock, int bx){
  using namespace scoutingRun3;

  int32_t ET(0), Eta(0), Phi(0), Iso(0);
  for (uint32_t i=0; i<6; i++) {
    ET = ((dataBlock[i] >> demux::shiftsEGamma::ET)  & demux::masksEGamma::ET);
    if (ET != 0) {
      Eta   = ((dataBlock[i] >> demux::shiftsEGamma::eta) & demux::masksEGamma::eta);
      Phi   = ((dataBlock[i] >> demux::shiftsEGamma::phi) & demux::masksEGamma::phi);
      Iso   = ((dataBlock[i] >> demux::shiftsEGamma::iso) & demux::masksEGamma::iso);

      if (Eta > 127) Eta = Eta - 256;

      ScEGamma eGamma(ET, Eta, Phi, Iso);
      orbitBufferEGammas_[bx].push_back(eGamma);
      nEGammasOrbit_ ++;
      
      if (debug_){
        std::cout << "E/g " << i << std::endl;
        std::cout << "  Raw: 0x" << std::hex << dataBlock[i] << std::dec << std::endl;
        std::cout << "  Et  [GeV/Hw]: " << demux::fEt(eGamma.hwEt())   << "/" << eGamma.hwEt()  << "\n";
        std::cout << "  Eta [rad/Hw]: " << demux::fEta(eGamma.hwEta()) << "/" << eGamma.hwEta() << "\n";
        std::cout << "  Phi [rad/Hw]: " << demux::fPhi(eGamma.hwPhi()) << "/" << eGamma.hwPhi() << "\n";
        std::cout << "  Iso [Hw]: " << eGamma.hwIso() << "\n";
      }
    }
  } // end link e/gammas unpacking loop
}

void ScCaloRawToDigi::unpackLinkTaus(uint32_t* dataBlock, int bx){
  using namespace scoutingRun3;

  int32_t ET(0), Eta(0), Phi(0), Iso(0);
  for (uint32_t i=0; i<6; i++) {
    ET = ((dataBlock[i] >> demux::shiftsTau::ET)  & demux::masksTau::ET);
    if (ET != 0) {
      Eta   = ((dataBlock[i] >> demux::shiftsTau::eta) & demux::masksTau::eta);
      Phi   = ((dataBlock[i] >> demux::shiftsTau::phi) & demux::masksTau::phi);
      Iso   = ((dataBlock[i] >> demux::shiftsTau::iso) & demux::masksTau::iso);

      if (Eta > 127) Eta = Eta - 256;

      ScTau tau(ET, Eta, Phi, Iso);
      orbitBufferTaus_[bx].push_back(tau);
      nTausOrbit_ ++;
      
      if (debug_){
        std::cout << "Tau " << i << std::endl;
        std::cout << "  Raw: 0x" << std::hex << dataBlock[i] << std::dec << std::endl;
        std::cout << "  Et  [GeV/Hw]: " << demux::fEt(tau.hwEt())   << "/" << tau.hwEt()  << "\n";
        std::cout << "  Eta [rad/Hw]: " << demux::fEta(tau.hwEta()) << "/" << tau.hwEta() << "\n";
        std::cout << "  Phi [rad/Hw]: " << demux::fPhi(tau.hwPhi()) << "/" << tau.hwPhi() << "\n";
        std::cout << "  Iso [Hw]: " << tau.hwIso() << "\n";
      }
    }
  } // end link taus unpacking loop
}

void ScCaloRawToDigi::unpackEtSums(uint32_t* dataBlock, int bx){
  using namespace scoutingRun3;

  int32_t ETEt(0), HTEt(0), ETmissEt(0), ETmissPhi(0), HTmissEt(0), HTmissPhi(0);

  // ET
  ETEt = ((dataBlock[0] >> demux::shiftsESums::ETEt) & demux::masksESums::ETEt);
  
  ScEtSum sumTotEt(ETEt, 0, l1t::EtSum::EtSumType::kTotalEt);
  orbitBufferEtSums_[bx].push_back(sumTotEt);

  // HT
  HTEt = ((dataBlock[1] >> demux::shiftsESums::HTEt) & demux::masksESums::HTEt);

  ScEtSum sumTotHt(HTEt, 0, l1t::EtSum::EtSumType::kTotalHt);
  orbitBufferEtSums_[bx].push_back(sumTotHt);

  // ETMiss
  ETmissEt  = ((dataBlock[2] >> demux::shiftsESums::ETmissEt)  & demux::masksESums::ETmissEt);
  ETmissPhi = ((dataBlock[2] >> demux::shiftsESums::ETmissPhi) & demux::masksESums::ETmissPhi);

  ScEtSum sumMissEt(ETmissEt, ETmissPhi, l1t::EtSum::EtSumType::kMissingEt);
  orbitBufferEtSums_[bx].push_back(sumMissEt);
  
  // HTMiss
  HTmissEt  = ((dataBlock[3] >> demux::shiftsESums::HTmissEt)  & demux::masksESums::HTmissEt);
  HTmissPhi = ((dataBlock[3] >> demux::shiftsESums::HTmissPhi) & demux::masksESums::HTmissPhi);

  ScEtSum sumMissHt(HTmissEt, HTmissPhi, l1t::EtSum::EtSumType::kMissingHt);
  orbitBufferEtSums_[bx].push_back(sumMissHt);

  nEtSumsOrbit_ += 4;

  if (debug_){
    std::cout << "Type: TotalET\n"
              << "  Raw: 0x" << std::hex << dataBlock[0] << std::dec << "\n"
              << "  Et [GeV/Hw]: " << demux::fEt(sumTotEt.hwEt()) << "/" << sumTotEt.hwEt()
              << std::endl;

    std::cout << "Type: TotalHT\n"
              << "  Raw: 0x" << std::hex << dataBlock[1] << std::dec << "\n"
              << "  Et [GeV/Hw]: " << demux::fEt(sumTotEt.hwEt()) << "/" << sumTotEt.hwEt()
              << std::endl;

    std::cout << "Type: ETMiss\n"
              << "  Raw: 0x" << std::hex << dataBlock[2] << std::dec << "\n"
              << "  Et  [GeV/Hw]: " << demux::fEt(sumMissEt.hwEt()) << "/" << sumMissEt.hwEt() << "\n"
              << "  Phi [Rad/Hw]: " << demux::fPhi(sumMissEt.hwPhi()) << "/" << sumMissEt.hwPhi()
              << std::endl;

    std::cout << "Type: HTMiss\n"
              << "  Raw: 0x" << std::hex << dataBlock[3] << std::dec << "\n"
              << "  Et  [GeV/Hw]: " << demux::fEt(sumMissHt.hwEt()) << "/" << sumMissHt.hwEt() << "\n"
              << "  Phi [Rad/Hw]: " << demux::fPhi(sumMissHt.hwPhi()) << "/" << sumMissHt.hwPhi()
              << std::endl;
  }
}

void ScCaloRawToDigi::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(ScCaloRawToDigi);
