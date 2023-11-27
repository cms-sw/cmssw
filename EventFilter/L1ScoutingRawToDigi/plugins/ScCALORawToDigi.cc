#include "EventFilter/L1ScoutingRawToDigi/plugins/ScCALORawToDigi.h"

ScCaloRawToDigi::ScCaloRawToDigi(const edm::ParameterSet& iConfig) {
  using namespace edm;
  using namespace l1ScoutingRun3;
  srcInputTag  = iConfig.getParameter<InputTag>("srcInputTag");
  debug_ = iConfig.getUntrackedParameter<bool>("debug", false);

  orbitBufferJets_    = std::vector<std::vector<ScJet>>(3565);
  orbitBufferEGammas_ = std::vector<std::vector<ScEGamma>>(3565);
  orbitBufferTaus_    = std::vector<std::vector<ScTau>>(3565);
  // orbitBufferEtSums_  = std::vector<std::vector<ScEtSum>>(3565);
  orbitBufferEtSums_  = std::vector<std::vector<ScBxSums>>(3565);

  nJetsOrbit_=0; nEGammasOrbit_=0; nTausOrbit_=0; nEtSumsOrbit_=0;

  produces<ScJetOrbitCollection>().setBranchAlias( "ScJetOrbitCollection" );
  produces<ScTauOrbitCollection>().setBranchAlias( "ScTauOrbitCollection" );
  produces<ScEGammaOrbitCollection>().setBranchAlias( "ScEGammaOrbitCollection" );
  // produces<ScEtSumOrbitCollection>().setBranchAlias( "ScEtSumOrbitCollection" );
  produces<ScBxSumsOrbitCollection>().setBranchAlias( "ScBxSumsOrbitCollection" );
  
  rawToken = consumes<SRDCollection>(srcInputTag);
  }

ScCaloRawToDigi::~ScCaloRawToDigi() {};

void ScCaloRawToDigi::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace l1ScoutingRun3;

  Handle<SRDCollection> ScoutingRawDataCollection;
  iEvent.getByToken( rawToken, ScoutingRawDataCollection );

  const FEDRawData& sourceRawData = ScoutingRawDataCollection->FEDData(SDSNumbering::CaloSDSID);
  size_t orbitSize = sourceRawData.size();

  std::unique_ptr<ScJetOrbitCollection> unpackedJets(new ScJetOrbitCollection);
  std::unique_ptr<ScTauOrbitCollection> unpackedTaus(new ScTauOrbitCollection);
  std::unique_ptr<ScEGammaOrbitCollection> unpackedEGammas(new ScEGammaOrbitCollection);
  // std::unique_ptr<ScEtSumOrbitCollection> unpackedEtSums(new ScEtSumOrbitCollection);
  std::unique_ptr<ScBxSumsOrbitCollection> unpackedEtSums(new ScBxSumsOrbitCollection);
  
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
  using namespace l1ScoutingRun3;

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
  using namespace l1ScoutingRun3;

  int32_t ET(0), Eta(0), Phi(0), Qual(0);
  for (uint32_t i=0; i<6; i++) {
    ET = ((dataBlock[i] >> demux::shiftsJet::ET)  & demux::masksJet::ET);
    
    if (ET != 0) {
      Eta  = ((dataBlock[i] >> demux::shiftsJet::eta) & demux::masksJet::eta);
      Phi  = ((dataBlock[i] >> demux::shiftsJet::phi) & demux::masksJet::phi);
      Qual = ((dataBlock[i] >> demux::shiftsJet::qual) & demux::masksJet::qual);

      if (Eta > 127) Eta = Eta - 256;

      ScJet jet(ET, Eta, Phi, Qual);
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
  using namespace l1ScoutingRun3;

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
  using namespace l1ScoutingRun3;

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
  using namespace l1ScoutingRun3;

  ScBxSums bxSums;

  int32_t ETEt(0), ETEttem(0), ETMinBiasHFP0(0); // ET
  int32_t HTEt(0), HTtowerCount(0), HTMinBiasHFM0(0); // HT
  int32_t ETmissEt(0), ETmissPhi(0), ETmissASYMET(0), ETmissMinBiasHFP1(0); //ETMiss
  int32_t HTmissEt(0), HTmissPhi(0), HTmissASYMHT(0), HTmissMinBiasHFM1(0); // HTMiss
  int32_t ETHFmissEt(0), ETHFmissPhi(0), ETHFmissASYMETHF(0), ETHFmissCENT(0); // ETHFMiss
  int32_t HTHFmissEt(0), HTHFmissPhi(0), HTHFmissASYMHTHF(0), HTHFmissCENT(0); // HTHFMiss
          
  // ET
  ETEt = ((dataBlock[0] >> demux::shiftsESums::ETEt) & demux::masksESums::ETEt);
  ETEttem       = ((dataBlock[0] >> demux::shiftsESums::ETEttem)     & demux::masksESums::ETEttem);
  ETMinBiasHFP0 = ((dataBlock[0] >> demux::shiftsESums::ETMinBiasHF) & demux::masksESums::ETMinBiasHF);

  // orbitBufferEtSums_[bx].emplace_back(ScEtSum(ETEt, 0, l1t::EtSum::EtSumType::kTotalEt));
  // orbitBufferEtSums_[bx].emplace_back(ScEtSum(ETEttem, 0, l1t::EtSum::EtSumType::kTotalEtEm));
  // orbitBufferEtSums_[bx].emplace_back(ScEtSum(ETMinBiasHFP0, 0, l1t::EtSum::EtSumType::kMinBiasHFP0));
  bxSums.setHwTotalEt(ETEt);
  bxSums.setHwTotalEtEm(ETEttem);
  bxSums.setMinBiasHFP0(ETMinBiasHFP0);

  // HT
  HTEt = ((dataBlock[1] >> demux::shiftsESums::HTEt) & demux::masksESums::HTEt);
  HTtowerCount = ((dataBlock[1] >> demux::shiftsESums::HTtowerCount) & demux::masksESums::HTtowerCount);
  HTMinBiasHFM0  = ((dataBlock[1] >> demux::shiftsESums::HTMinBiasHF)  & demux::masksESums::HTMinBiasHF);

  // orbitBufferEtSums_[bx].emplace_back(ScEtSum(HTEt, 0, l1t::EtSum::EtSumType::kTotalHt));
  // orbitBufferEtSums_[bx].emplace_back(ScEtSum(HTtowerCount, 0, l1t::EtSum::EtSumType::kTowerCount));
  // orbitBufferEtSums_[bx].emplace_back(ScEtSum(HTMinBiasHFM0, 0, l1t::EtSum::EtSumType::kMinBiasHFM0));

  bxSums.setHwTotalHt(HTEt);
  bxSums.setTowerCount(HTtowerCount);
  bxSums.setMinBiasHFM0(HTMinBiasHFM0);

  // ETMiss
  ETmissEt  = ((dataBlock[2] >> demux::shiftsESums::ETmissEt)  & demux::masksESums::ETmissEt);
  ETmissPhi = ((dataBlock[2] >> demux::shiftsESums::ETmissPhi) & demux::masksESums::ETmissPhi);
  ETmissASYMET      = ((dataBlock[2] >> demux::shiftsESums::ETmissASYMET)    & demux::masksESums::ETmissASYMET);
  ETmissMinBiasHFP1 = ((dataBlock[2] >> demux::shiftsESums::ETmissMinBiasHF) & demux::masksESums::ETmissMinBiasHF);

  // orbitBufferEtSums_[bx].emplace_back(ScEtSum(ETmissEt, ETmissPhi, l1t::EtSum::EtSumType::kMissingEt));
  // orbitBufferEtSums_[bx].emplace_back(ScEtSum(ETmissASYMET, 0, l1t::EtSum::EtSumType::kAsymEt));
  // orbitBufferEtSums_[bx].emplace_back(ScEtSum(ETmissMinBiasHFP1, 0, l1t::EtSum::EtSumType::kMinBiasHFP1));

  bxSums.setHwMissEt(ETmissEt);
  bxSums.setHwMissEtPhi(ETmissPhi);
  bxSums.setHwAsymEt(ETmissASYMET);
  bxSums.setMinBiasHFP1(ETmissMinBiasHFP1);
  
  // HTMiss
  HTmissEt  = ((dataBlock[3] >> demux::shiftsESums::HTmissEt)  & demux::masksESums::HTmissEt);
  HTmissPhi = ((dataBlock[3] >> demux::shiftsESums::HTmissPhi) & demux::masksESums::HTmissPhi);
  HTmissASYMHT      = ((dataBlock[3] >> demux::shiftsESums::HTmissASYMHT)    & demux::masksESums::HTmissASYMHT);
  HTmissMinBiasHFM1 = ((dataBlock[3] >> demux::shiftsESums::HTmissMinBiasHF) & demux::masksESums::HTmissMinBiasHF);

  // orbitBufferEtSums_[bx].emplace_back(ScEtSum(HTmissEt, HTmissPhi, l1t::EtSum::EtSumType::kMissingHt));
  // orbitBufferEtSums_[bx].emplace_back(ScEtSum(HTmissASYMHT, 0, l1t::EtSum::EtSumType::kAsymHt));
  // orbitBufferEtSums_[bx].emplace_back(ScEtSum(HTmissMinBiasHFM1, 0, l1t::EtSum::EtSumType::kMinBiasHFM1));

  std::cout << "BX " << bx << ", Frame 3:  " << std::endl;
  std::cout << "  Raw: 0x" << std::hex << dataBlock[3] << std::dec << "\n"
            << "  MHTEt: " << HTmissEt << ",  MHTEt PHI: " << HTmissPhi << std::endl;

  bxSums.setHwMissHt(HTmissEt);
  bxSums.setHwMissHtPhi(HTmissPhi);
  bxSums.setHwAsymHt(HTmissASYMHT);
  bxSums.setMinBiasHFM1(HTmissMinBiasHFM1);

  // ETHFMiss
  ETHFmissEt       = ((dataBlock[4] >> demux::shiftsESums::ETHFmissEt)       & demux::masksESums::ETHFmissEt);
  ETHFmissPhi      = ((dataBlock[4] >> demux::shiftsESums::ETHFmissPhi)      & demux::masksESums::ETHFmissPhi);
  ETHFmissASYMETHF = ((dataBlock[4] >> demux::shiftsESums::ETHFmissASYMETHF) & demux::masksESums::ETHFmissASYMETHF);
  ETHFmissCENT     = ((dataBlock[4] >> demux::shiftsESums::ETHFmissCENT)     & demux::masksESums::ETHFmissCENT);

  // orbitBufferEtSums_[bx].emplace_back(ScEtSum(ETHFmissEt, ETHFmissPhi, l1t::EtSum::EtSumType::kMissingEtHF));
  // orbitBufferEtSums_[bx].emplace_back(ScEtSum(ETHFmissASYMETHF, 0, l1t::EtSum::EtSumType::kAsymEtHF));

  bxSums.setHwMissEtHF(ETHFmissEt);
  bxSums.setHwMissEtHFPhi(ETHFmissPhi);
  bxSums.setHwAsymEtHF(ETHFmissASYMETHF);

  // HTHFMiss
  HTHFmissEt        = ((dataBlock[5] >> demux::shiftsESums::ETHFmissEt)       & demux::masksESums::ETHFmissEt);
  HTHFmissPhi       = ((dataBlock[5] >> demux::shiftsESums::ETHFmissPhi)      & demux::masksESums::ETHFmissPhi);
  HTHFmissASYMHTHF  = ((dataBlock[5] >> demux::shiftsESums::ETHFmissASYMETHF) & demux::masksESums::ETHFmissASYMETHF);
  HTHFmissCENT      = ((dataBlock[5] >> demux::shiftsESums::ETHFmissCENT)     & demux::masksESums::ETHFmissCENT);

  // orbitBufferEtSums_[bx].emplace_back(ScEtSum(HTHFmissEt, HTHFmissPhi, l1t::EtSum::EtSumType::kMissingHtHF));
  // orbitBufferEtSums_[bx].emplace_back(ScEtSum(HTHFmissASYMHTHF, 0, l1t::EtSum::EtSumType::kAsymHtHF));
  // orbitBufferEtSums_[bx].emplace_back(ScEtSum((HTHFmissCENT<<4) + ETHFmissCENT, 0, l1t::EtSum::EtSumType::kCentrality));

  bxSums.setHwMissHtHF(HTHFmissEt);
  bxSums.setHwMissHtHFPhi(HTHFmissPhi);
  bxSums.setHwAsymHtHF(HTHFmissASYMHTHF);
  bxSums.setCentrality((HTHFmissCENT<<4) + ETHFmissCENT);

  // nEtSumsOrbit_ += 17;
  nEtSumsOrbit_ += 1;

  orbitBufferEtSums_[bx].push_back(bxSums);

  if (debug_){
    // std::cout << "Type: TotalET\n"
    //           << "  Raw: 0x" << std::hex << dataBlock[0] << std::dec << "\n"
    //           << "  Et [GeV/Hw]: " << demux::fEt(sumTotEt.hwEt()) << "/" << sumTotEt.hwEt()
    //           << std::endl;

    // std::cout << "Type: TotalHT\n"
    //           << "  Raw: 0x" << std::hex << dataBlock[1] << std::dec << "\n"
    //           << "  Et [GeV/Hw]: " << demux::fEt(sumTotHt.hwEt()) << "/" << sumTotHt.hwEt()
    //           << std::endl;

    // std::cout << "Type: ETMiss\n"
    //           << "  Raw: 0x" << std::hex << dataBlock[2] << std::dec << "\n"
    //           << "  Et  [GeV/Hw]: " << demux::fEt(sumMissEt.hwEt()) << "/" << sumMissEt.hwEt() << "\n"
    //           << "  Phi [Rad/Hw]: " << demux::fPhi(sumMissEt.hwPhi()) << "/" << sumMissEt.hwPhi()
    //           << std::endl;

    // std::cout << "Type: HTMiss\n"
    //           << "  Raw: 0x" << std::hex << dataBlock[3] << std::dec << "\n"
    //           << "  Et  [GeV/Hw]: " << demux::fEt(sumMissHt.hwEt()) << "/" << sumMissHt.hwEt() << "\n"
    //           << "  Phi [Rad/Hw]: " << demux::fPhi(sumMissHt.hwPhi()) << "/" << sumMissHt.hwPhi()
    //           << std::endl;
  }
}

void ScCaloRawToDigi::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(ScCaloRawToDigi);
