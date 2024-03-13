#include "EventFilter/L1ScoutingRawToDigi/plugins/ScCALORawToDigi.h"

ScCaloRawToDigi::ScCaloRawToDigi(const edm::ParameterSet& iConfig) {
  using namespace edm;
  using namespace l1ScoutingRun3;
  srcInputTag_ = iConfig.getParameter<InputTag>("srcInputTag");
  enableAllSums_ = iConfig.getParameter<bool>("enableAllSums");
  debug_ = iConfig.getUntrackedParameter<bool>("debug", false);
  dataSourceConfig_ = iConfig.getParameter<edm::ParameterSet>("dataSource");
  rawToken_ = consumes<SDSRawDataCollection>(srcInputTag_);

  orbitBufferJets_ = std::vector<std::vector<Jet>>(3565);
  orbitBufferEGammas_ = std::vector<std::vector<EGamma>>(3565);
  orbitBufferTaus_ = std::vector<std::vector<Tau>>(3565);
  orbitBufferEtSums_ = std::vector<std::vector<BxSums>>(3565);

  nJetsOrbit_ = 0;
  nEGammasOrbit_ = 0;
  nTausOrbit_ = 0;
  nEtSumsOrbit_ = 0;

  produces<JetOrbitCollection>().setBranchAlias("JetOrbitCollection");
  produces<TauOrbitCollection>().setBranchAlias("TauOrbitCollection");
  produces<EGammaOrbitCollection>().setBranchAlias("EGammaOrbitCollection");
  produces<BxSumsOrbitCollection>().setBranchAlias("BxSumsOrbitCollection");
}

ScCaloRawToDigi::~ScCaloRawToDigi(){};

void ScCaloRawToDigi::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace l1ScoutingRun3;

  Handle<SDSRawDataCollection> ScoutingRawDataCollection;
  iEvent.getByToken(rawToken_, ScoutingRawDataCollection);

  std::unique_ptr<JetOrbitCollection> unpackedJets(new JetOrbitCollection);
  std::unique_ptr<TauOrbitCollection> unpackedTaus(new TauOrbitCollection);
  std::unique_ptr<EGammaOrbitCollection> unpackedEGammas(new EGammaOrbitCollection);
  std::unique_ptr<BxSumsOrbitCollection> unpackedEtSums(new BxSumsOrbitCollection);

  // reset counters
  nJetsOrbit_ = 0;
  nEGammasOrbit_ = 0;
  nTausOrbit_ = 0;
  nEtSumsOrbit_ = 0;

  std::string dataSourceMode = dataSourceConfig_.getParameter<std::string>("dataSourceMode");
  if (dataSourceMode == "DMA") {
    // Packet from DMA contains all the objects
    int sourceId = dataSourceConfig_.getParameter<int>("sourceId"); 
    if (sourceId!=SDSNumbering::CaloSDSID) edm::LogWarning("ScCaloRawToDIgi::produce")
                                            << "Provided an unexpected source ID: "
                                            << sourceId << "/" << SDSNumbering::CaloSDSID
                                            << " [provided/expected]";
    unpackOrbitFromDMA(ScoutingRawDataCollection, sourceId);
  }
  else if (dataSourceMode == "TCP") {
    // unpack jets
    jetSourceIdList_ = dataSourceConfig_.getParameter<std::vector<int>>("jetSourceIdList");
    unpackTcpData(ScoutingRawDataCollection,
                  jetSourceIdList_, CaloObjectType::Jet);

    // unpack e/gamma
    eGammaSourceIdList_ = dataSourceConfig_.getParameter<std::vector<int>>("eGammaSourceIdList");
    unpackTcpData(ScoutingRawDataCollection,
                  eGammaSourceIdList_, CaloObjectType::EGamma);

    // unpack taus
    tauSourceIdList_ = dataSourceConfig_.getParameter<std::vector<int>>("tauSourceIdList");
    unpackTcpData(ScoutingRawDataCollection,
                  tauSourceIdList_, CaloObjectType::Tau);

    // unpack et sums
    etSumSourceIdList_ = dataSourceConfig_.getParameter<std::vector<int>>("etSumSourceIdList");
    unpackTcpData(ScoutingRawDataCollection,
                  etSumSourceIdList_, CaloObjectType::EtSum);
  } else {
    throw cms::Exception("ScCaloRawToDIgi::produce") 
            << "Unknown data source mode. Use DMA or TCP(default).";
  }

  // fill orbit collection and clear the Bx buffer vector
  unpackedJets->fillAndClear(orbitBufferJets_, nJetsOrbit_);
  unpackedEGammas->fillAndClear(orbitBufferEGammas_, nEGammasOrbit_);
  unpackedTaus->fillAndClear(orbitBufferTaus_, nTausOrbit_);
  unpackedEtSums->fillAndClear(orbitBufferEtSums_, nEtSumsOrbit_);

  // store collections in the event
  iEvent.put(std::move(unpackedJets));
  iEvent.put(std::move(unpackedTaus));
  iEvent.put(std::move(unpackedEGammas));
  iEvent.put(std::move(unpackedEtSums));
}

void ScCaloRawToDigi::unpackOrbitFromDMA(edm::Handle<SDSRawDataCollection> &ScoutingRawDataCollection,
                                         int sourceId) {
  using namespace l1ScoutingRun3;

  const FEDRawData& sourceRawData = ScoutingRawDataCollection->FEDData(sourceId);
  if ((sourceRawData.size() == 0) && debug_) {
    std::cout << "No raw data for CALO DMA source ID=" << sourceId << std::endl;
  }

  // get orbit size and raw data
  size_t len = sourceRawData.size();
  const unsigned char* buf = sourceRawData.data();

  size_t pos = 0;

  while (pos < len) {
    assert(pos + sizeof(demux::dmaBlock) <= len);

    demux::dmaBlock* bl = (demux::dmaBlock*)(buf + pos);
    pos += sizeof(demux::dmaBlock);

    assert(pos <= len);
    uint32_t orbit = bl->orbit & 0x7FFFFFFF;
    uint32_t bx = bl->bx;

    if (debug_) {
      std::cout << "CALO Orbit " << orbit << ", BX -> " << bx << std::endl;
    }

    // unpack jets from first link
    if (debug_)
      std::cout << "--- Jets link 1 ---\n";
    unpackJets(bl->jet1, bx, 6);

    // unpack jets from second link
    if (debug_)
      std::cout << "--- Jets link 2 ---\n";
    unpackJets(bl->jet2, bx, 6);

    // unpack eg from first link
    if (debug_)
      std::cout << "--- E/g link 1 ---\n";
    unpackEGammas(bl->egamma1, bx, 6);

    // unpack eg from second link link
    if (debug_)
      std::cout << "--- E/g link 2 ---\n";
    unpackEGammas(bl->egamma2, bx, 6);

    // unpack taus from first link
    if (debug_)
      std::cout << "--- Taus link 1 ---\n";
    unpackTaus(bl->tau1, bx, 6);

    // unpack taus from second link
    if (debug_)
      std::cout << "--- Taus link 2 ---\n";
    unpackTaus(bl->tau2, bx, 6);

    // unpack et sums
    if (debug_)
      std::cout << "--- Sums ---\n";
    unpackEtSums(bl->sum, bx);

  }  // end of bx objects
}

void ScCaloRawToDigi::unpackTcpData(edm::Handle<SDSRawDataCollection> &ScoutingRawDataCollection,
                                    std::vector<int> sourceList,
                                    CaloObjectType dataType){
  using namespace l1ScoutingRun3;
  for (const int &sourceId: sourceList){
    if ((sourceId<SDSNumbering::CaloTCPMinSDSID)||(sourceId>SDSNumbering::CaloTCPMaxSDSID)) {
      edm::LogWarning("ScCaloRawToDIgi::unpackTCPData")
                      << "Source ID outside the expected range " << sourceId 
                      << "/[" << SDSNumbering::CaloTCPMinSDSID << "-" << SDSNumbering::CaloTCPMaxSDSID << "]"
                      << " [provided/expected range]";
    }
    const FEDRawData& sourceRawData = ScoutingRawDataCollection->FEDData(sourceId);
    size_t orbitSize = sourceRawData.size();

    if ((sourceRawData.size() == 0) && debug_) {
      //std::cout << "No raw data for calo " << CaloObjTypeTxt[dataType] << ", TCP source ID=" << sourceId << std::endl;
      std::cout << "No raw data for calo TCP source ID=" << sourceId << std::endl;
    }

    unpackOrbitFromTCP(sourceRawData.data(), orbitSize, dataType);
  }
}

void ScCaloRawToDigi::unpackOrbitFromTCP(const unsigned char* buf, size_t len, CaloObjectType dataType) {
  using namespace l1ScoutingRun3;
  size_t pos = 0;

  while (pos < len) {
    // frame header is present
    assert(pos + 4 <= len);
    
    // unpack calo sums block
    if (dataType==CaloObjectType::EtSum){
      demux::caloSumTcpBlock* bl = (demux::caloSumTcpBlock*)(buf + pos);
      pos += sizeof(demux::caloSumTcpBlock);
      assert(pos <= len);
      if (debug_)
        std::cout << "Sums BX -> " << bl->bx << std::endl;
      unpackEtSums(bl->sum, bl->bx);
    } else {
      // unpack jet/eg/tau
      demux::caloObjTcpBlock* bl = (demux::caloObjTcpBlock*)(buf + pos);
      int nObj = (bl->header) & 0xff;
      pos += 12 + nObj*4;

      switch (dataType) {
        case CaloObjectType::Jet:
          if (debug_)
            std::cout << "Jets BX -> " << bl->bx << std::endl;
          unpackJets(bl->obj, bl->bx, nObj);
          break;

        case CaloObjectType::EGamma:
          if (debug_)
            std::cout << "E/Gammas BX -> " << bl->bx << std::endl;
          unpackEGammas(bl->obj, bl->bx, nObj);
          break;

        case CaloObjectType::Tau:
          if (debug_)
            std::cout << "Taus BX -> " << bl->bx << std::endl;
          unpackTaus(bl->obj, bl->bx, nObj);
          break;

        default:
          throw cms::Exception("ScCaloRawToDigi::unpackOrbitFromTCP") 
                << "Unknown data type.";
          break;
      }

    } // unpack sums and calo objects
    
  }  // end of bx objects
}

void ScCaloRawToDigi::unpackJets(uint32_t* dataBlock, int bx, int nObjets) {
  using namespace l1ScoutingRun3;

  int32_t ET(0), Eta(0), Phi(0), Qual(0);
  for (int i = 0; i < nObjets; i++) {
    ET = ((dataBlock[i] >> demux::shiftsJet::ET) & demux::masksJet::ET);

    if (ET != 0) {
      Eta = ((dataBlock[i] >> demux::shiftsJet::eta) & demux::masksJet::eta);
      Phi = ((dataBlock[i] >> demux::shiftsJet::phi) & demux::masksJet::phi);
      Qual = ((dataBlock[i] >> demux::shiftsJet::qual) & demux::masksJet::qual);

      if (Eta > 127)
        Eta = Eta - 256;

      Jet jet(ET, Eta, Phi, Qual);
      orbitBufferJets_[bx].push_back(jet);
      nJetsOrbit_++;

      if (debug_) {
        std::cout << "Jet " << i << std::endl;
        std::cout << "  Raw: 0x" << std::hex << dataBlock[i] << std::dec << std::endl;
        printJet(jet);
      }
    }
  }  // end link jets unpacking loop
}

void ScCaloRawToDigi::unpackEGammas(uint32_t* dataBlock, int bx, int nObjets) {
  using namespace l1ScoutingRun3;

  int32_t ET(0), Eta(0), Phi(0), Iso(0);
  for (int i = 0; i < nObjets; i++) {
    ET = ((dataBlock[i] >> demux::shiftsEGamma::ET) & demux::masksEGamma::ET);
    if (ET != 0) {
      Eta = ((dataBlock[i] >> demux::shiftsEGamma::eta) & demux::masksEGamma::eta);
      Phi = ((dataBlock[i] >> demux::shiftsEGamma::phi) & demux::masksEGamma::phi);
      Iso = ((dataBlock[i] >> demux::shiftsEGamma::iso) & demux::masksEGamma::iso);

      if (Eta > 127)
        Eta = Eta - 256;

      EGamma eGamma(ET, Eta, Phi, Iso);
      orbitBufferEGammas_[bx].push_back(eGamma);
      nEGammasOrbit_++;

      if (debug_) {
        std::cout << "E/g " << i << std::endl;
        std::cout << "  Raw: 0x" << std::hex << dataBlock[i] << std::dec << std::endl;
        printEGamma(eGamma);
      }
    }
  }  // end link e/gammas unpacking loop
}

void ScCaloRawToDigi::unpackTaus(uint32_t* dataBlock, int bx, int nObjets) {
  using namespace l1ScoutingRun3;

  int32_t ET(0), Eta(0), Phi(0), Iso(0);
  for (int i = 0; i < nObjets; i++) {
    ET = ((dataBlock[i] >> demux::shiftsTau::ET) & demux::masksTau::ET);
    if (ET != 0) {
      Eta = ((dataBlock[i] >> demux::shiftsTau::eta) & demux::masksTau::eta);
      Phi = ((dataBlock[i] >> demux::shiftsTau::phi) & demux::masksTau::phi);
      Iso = ((dataBlock[i] >> demux::shiftsTau::iso) & demux::masksTau::iso);

      if (Eta > 127)
        Eta = Eta - 256;

      Tau tau(ET, Eta, Phi, Iso);
      orbitBufferTaus_[bx].push_back(tau);
      nTausOrbit_++;

      if (debug_) {
        std::cout << "Tau " << i << std::endl;
        std::cout << "  Raw: 0x" << std::hex << dataBlock[i] << std::dec << std::endl;
        printTau(tau);
      }
    }
  }  // end link taus unpacking loop
}

void ScCaloRawToDigi::unpackEtSums(uint32_t* dataBlock, int bx) {
  using namespace l1ScoutingRun3;

  BxSums bxSums;

  int32_t ETEt(0), ETEttem(0), ETMinBiasHFP0(0);                                // ET block
  int32_t HTEt(0), HTtowerCount(0), HTMinBiasHFM0(0);                           // HT block
  int32_t ETmissEt(0), ETmissPhi(0), ETmissASYMET(0), ETmissMinBiasHFP1(0);     // ETMiss block
  int32_t HTmissEt(0), HTmissPhi(0), HTmissASYMHT(0), HTmissMinBiasHFM1(0);     // HTMiss block
  int32_t ETHFmissEt(0), ETHFmissPhi(0), ETHFmissASYMETHF(0), ETHFmissCENT(0);  // ETHFMiss block
  int32_t HTHFmissEt(0), HTHFmissPhi(0), HTHFmissASYMHTHF(0), HTHFmissCENT(0);  // HTHFMiss block

  // ET block
  ETEt = ((dataBlock[0] >> demux::shiftsESums::ETEt) & demux::masksESums::ETEt);
  ETEttem = ((dataBlock[0] >> demux::shiftsESums::ETEttem) & demux::masksESums::ETEttem);

  bxSums.setHwTotalEt(ETEt);
  bxSums.setHwTotalEtEm(ETEttem);

  // HT block
  HTEt = ((dataBlock[1] >> demux::shiftsESums::HTEt) & demux::masksESums::HTEt);

  bxSums.setHwTotalHt(HTEt);

  // ETMiss block
  ETmissEt = ((dataBlock[2] >> demux::shiftsESums::ETmissEt) & demux::masksESums::ETmissEt);
  ETmissPhi = ((dataBlock[2] >> demux::shiftsESums::ETmissPhi) & demux::masksESums::ETmissPhi);

  if (ETmissEt > 0) {
    bxSums.setHwMissEt(ETmissEt);
    bxSums.setHwMissEtPhi(ETmissPhi);
  }

  // HTMiss block
  HTmissEt = ((dataBlock[3] >> demux::shiftsESums::HTmissEt) & demux::masksESums::HTmissEt);
  HTmissPhi = ((dataBlock[3] >> demux::shiftsESums::HTmissPhi) & demux::masksESums::HTmissPhi);

  if (HTmissEt > 0) {
    bxSums.setHwMissHt(HTmissEt);
    bxSums.setHwMissHtPhi(HTmissPhi);
  }

  // ETHFMiss block
  ETHFmissEt = ((dataBlock[4] >> demux::shiftsESums::ETHFmissEt) & demux::masksESums::ETHFmissEt);
  ETHFmissPhi = ((dataBlock[4] >> demux::shiftsESums::ETHFmissPhi) & demux::masksESums::ETHFmissPhi);

  if (ETHFmissEt > 0) {
    bxSums.setHwMissEtHF(ETHFmissEt);
    bxSums.setHwMissEtHFPhi(ETHFmissPhi);
  }

  // HTHFMiss block
  HTHFmissEt = ((dataBlock[5] >> demux::shiftsESums::ETHFmissEt) & demux::masksESums::ETHFmissEt);
  HTHFmissPhi = ((dataBlock[5] >> demux::shiftsESums::ETHFmissPhi) & demux::masksESums::ETHFmissPhi);

  if (HTHFmissEt > 0) {
    bxSums.setHwMissHtHF(HTHFmissEt);
    bxSums.setHwMissHtHFPhi(HTHFmissPhi);
  }

  // Insert additional sums
  if (enableAllSums_) {
    // ET block
    ETMinBiasHFP0 = ((dataBlock[0] >> demux::shiftsESums::ETMinBiasHF) & demux::masksESums::ETMinBiasHF);
    bxSums.setMinBiasHFP0(ETMinBiasHFP0);

    // HT block
    HTtowerCount = ((dataBlock[1] >> demux::shiftsESums::HTtowerCount) & demux::masksESums::HTtowerCount);
    HTMinBiasHFM0 = ((dataBlock[1] >> demux::shiftsESums::HTMinBiasHF) & demux::masksESums::HTMinBiasHF);

    bxSums.setTowerCount(HTtowerCount);
    bxSums.setMinBiasHFM0(HTMinBiasHFM0);

    // ET Miss block
    ETmissASYMET = ((dataBlock[2] >> demux::shiftsESums::ETmissASYMET) & demux::masksESums::ETmissASYMET);
    ETmissMinBiasHFP1 = ((dataBlock[2] >> demux::shiftsESums::ETmissMinBiasHF) & demux::masksESums::ETmissMinBiasHF);
    bxSums.setHwAsymEt(ETmissASYMET);
    bxSums.setMinBiasHFP1(ETmissMinBiasHFP1);

    // HT Miss block
    HTmissASYMHT = ((dataBlock[3] >> demux::shiftsESums::HTmissASYMHT) & demux::masksESums::HTmissASYMHT);
    HTmissMinBiasHFM1 = ((dataBlock[3] >> demux::shiftsESums::HTmissMinBiasHF) & demux::masksESums::HTmissMinBiasHF);

    bxSums.setHwAsymHt(HTmissASYMHT);
    bxSums.setMinBiasHFM1(HTmissMinBiasHFM1);

    // ETHFMiss
    ETHFmissASYMETHF = ((dataBlock[4] >> demux::shiftsESums::ETHFmissASYMETHF) & demux::masksESums::ETHFmissASYMETHF);
    ETHFmissCENT = ((dataBlock[4] >> demux::shiftsESums::ETHFmissCENT) & demux::masksESums::ETHFmissCENT);

    bxSums.setHwAsymEtHF(ETHFmissASYMETHF);

    // HTHFMiss
    HTHFmissASYMHTHF = ((dataBlock[5] >> demux::shiftsESums::ETHFmissASYMETHF) & demux::masksESums::ETHFmissASYMETHF);
    HTHFmissCENT = ((dataBlock[5] >> demux::shiftsESums::ETHFmissCENT) & demux::masksESums::ETHFmissCENT);

    bxSums.setHwAsymHtHF(HTHFmissASYMHTHF);
    bxSums.setCentrality((HTHFmissCENT << 4) + ETHFmissCENT);
  }

  orbitBufferEtSums_[bx].push_back(bxSums);
  nEtSumsOrbit_ += 1;

  if (debug_) {
    std::cout << "Raw frames:\n";
    for (int frame = 0; frame < 6; frame++) { 
      std::cout << "  frame " << frame << ": 0x" << std::hex << dataBlock[frame] << std::dec << std::endl;
    }
    printBxSums(bxSums);
  }
}

void ScCaloRawToDigi::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("srcInputTag", edm::InputTag("rawDataCollector"));
  {
    edm::ParameterSetDescription dataSource;
    dataSource.add<std::string>("dataSourceMode", std::string("TCP"));
    dataSource.add<std::vector<int>>("jetSourceIdList", std::vector<int>({22}));
    dataSource.add<std::vector<int>>("eGammaSourceIdList", std::vector<int>({23}));
    dataSource.add<std::vector<int>>("tauSourceIdList", std::vector<int>({25}));
    dataSource.add<std::vector<int>>("etSumSourceIdList", std::vector<int>({24}));
    desc.add("dataSource", dataSource);
  }
  desc.add<bool>("enableAllSums", true);
  desc.addUntracked<bool>("debug", false);
  descriptions.add("ScCaloRawToDigi", desc);
}

DEFINE_FWK_MODULE(ScCaloRawToDigi);
