#include "EventFilter/L1ScoutingRawToDigi/plugins/ScCALORawToDigi.h"

ScCaloRawToDigi::ScCaloRawToDigi(const edm::ParameterSet& iConfig) {
  srcInputTag_ = iConfig.getParameter<edm::InputTag>("srcInputTag");
  enableAllSums_ = iConfig.getParameter<bool>("enableAllSums");
  dataSourceConfig_ = iConfig.getParameter<edm::ParameterSet>("dataSource");
  rawToken_ = consumes<SDSRawDataCollection>(srcInputTag_);

  orbitBufferJets_ = std::vector<std::vector<l1ScoutingRun3::Jet>>(3565);
  orbitBufferEGammas_ = std::vector<std::vector<l1ScoutingRun3::EGamma>>(3565);
  orbitBufferTaus_ = std::vector<std::vector<l1ScoutingRun3::Tau>>(3565);
  orbitBufferEtSums_ = std::vector<std::vector<l1ScoutingRun3::BxSums>>(3565);

  nJetsOrbit_ = 0;
  nEGammasOrbit_ = 0;
  nTausOrbit_ = 0;
  nEtSumsOrbit_ = 0;

  produces<l1ScoutingRun3::JetOrbitCollection>("Jet").setBranchAlias("JetOrbitCollection");
  produces<l1ScoutingRun3::TauOrbitCollection>("Tau").setBranchAlias("TauOrbitCollection");
  produces<l1ScoutingRun3::EGammaOrbitCollection>("EGamma").setBranchAlias("EGammaOrbitCollection");
  produces<l1ScoutingRun3::BxSumsOrbitCollection>("EtSum").setBranchAlias("BxSumsOrbitCollection");
}

ScCaloRawToDigi::~ScCaloRawToDigi() {}

void ScCaloRawToDigi::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<SDSRawDataCollection> ScoutingRawDataCollection;
  iEvent.getByToken(rawToken_, ScoutingRawDataCollection);

  std::unique_ptr<l1ScoutingRun3::JetOrbitCollection> unpackedJets(new l1ScoutingRun3::JetOrbitCollection);
  std::unique_ptr<l1ScoutingRun3::TauOrbitCollection> unpackedTaus(new l1ScoutingRun3::TauOrbitCollection);
  std::unique_ptr<l1ScoutingRun3::EGammaOrbitCollection> unpackedEGammas(new l1ScoutingRun3::EGammaOrbitCollection);
  std::unique_ptr<l1ScoutingRun3::BxSumsOrbitCollection> unpackedEtSums(new l1ScoutingRun3::BxSumsOrbitCollection);

  // reset counters
  nJetsOrbit_ = 0;
  nEGammasOrbit_ = 0;
  nTausOrbit_ = 0;
  nEtSumsOrbit_ = 0;

  std::string dataSourceMode = dataSourceConfig_.getParameter<std::string>("dataSourceMode");
  if (dataSourceMode == "DMA") {
    // Packet from DMA contains all the objects
    int sourceId = dataSourceConfig_.getParameter<int>("dmaSourceId");
    if (sourceId != SDSNumbering::CaloSDSID)
      edm::LogWarning("ScCaloRawToDIgi::produce") << "Provided an unexpected source ID: " << sourceId << "/"
                                                  << SDSNumbering::CaloSDSID << " [provided/expected]";
    unpackOrbitFromDMA(ScoutingRawDataCollection, sourceId);
  } else if (dataSourceMode == "TCP") {
    // unpack jets
    jetSourceIdList_ = dataSourceConfig_.getParameter<std::vector<int>>("jetSourceIdList");
    unpackTcpData(ScoutingRawDataCollection, jetSourceIdList_, CaloObjectType::Jet);

    // unpack e/gamma
    eGammaSourceIdList_ = dataSourceConfig_.getParameter<std::vector<int>>("eGammaSourceIdList");
    unpackTcpData(ScoutingRawDataCollection, eGammaSourceIdList_, CaloObjectType::EGamma);

    // unpack taus
    tauSourceIdList_ = dataSourceConfig_.getParameter<std::vector<int>>("tauSourceIdList");
    unpackTcpData(ScoutingRawDataCollection, tauSourceIdList_, CaloObjectType::Tau);

    // unpack et sums
    etSumSourceIdList_ = dataSourceConfig_.getParameter<std::vector<int>>("etSumSourceIdList");
    unpackTcpData(ScoutingRawDataCollection, etSumSourceIdList_, CaloObjectType::EtSum);
  } else {
    throw cms::Exception("ScCaloRawToDIgi::produce") << "Unknown data source mode. Use DMA or TCP(default).";
  }

  // fill orbit collection and clear the Bx buffer vector
  unpackedJets->fillAndClear(orbitBufferJets_, nJetsOrbit_);
  unpackedEGammas->fillAndClear(orbitBufferEGammas_, nEGammasOrbit_);
  unpackedTaus->fillAndClear(orbitBufferTaus_, nTausOrbit_);
  unpackedEtSums->fillAndClear(orbitBufferEtSums_, nEtSumsOrbit_);

  // store collections in the event
  iEvent.put(std::move(unpackedJets), "Jet");
  iEvent.put(std::move(unpackedTaus), "Tau");
  iEvent.put(std::move(unpackedEGammas), "EGamma");
  iEvent.put(std::move(unpackedEtSums), "EtSum");
}

void ScCaloRawToDigi::unpackOrbitFromDMA(edm::Handle<SDSRawDataCollection>& ScoutingRawDataCollection, int sourceId) {
  const FEDRawData& sourceRawData = ScoutingRawDataCollection->FEDData(sourceId);
  if (sourceRawData.size() == 0) {
    LogDebug("L1Scout") << "No raw data for CALO DMA source ID=" << sourceId;
  }

  // get orbit size and raw data
  size_t len = sourceRawData.size();
  const unsigned char* buf = sourceRawData.data();

  size_t pos = 0;

  while (pos < len) {
    assert(pos + sizeof(l1ScoutingRun3::demux::dmaBlock) <= len);

    l1ScoutingRun3::demux::dmaBlock* bl = (l1ScoutingRun3::demux::dmaBlock*)(buf + pos);
    pos += sizeof(l1ScoutingRun3::demux::dmaBlock);

    assert(pos <= len);
    uint32_t orbit = bl->orbit & 0x7FFFFFFF;
    uint32_t bx = bl->bx;

    LogDebug("L1Scout") << "CALO Orbit " << orbit << ", BX -> " << bx;

    // unpack jets from first link
    LogDebug("L1Scout") << "--- Jets link 1 ---\n";
    unpackJets(bl->jet1, bx, 6);

    // unpack jets from second link
    LogDebug("L1Scout") << "--- Jets link 2 ---\n";
    unpackJets(bl->jet2, bx, 6);

    // unpack eg from first link
    LogDebug("L1Scout") << "--- E/g link 1 ---\n";
    unpackEGammas(bl->egamma1, bx, 6);

    // unpack eg from second link link
    LogDebug("L1Scout") << "--- E/g link 2 ---\n";
    unpackEGammas(bl->egamma2, bx, 6);

    // unpack taus from first link
    LogDebug("L1Scout") << "--- Taus link 1 ---\n";
    unpackTaus(bl->tau1, bx, 6);

    // unpack taus from second link
    LogDebug("L1Scout") << "--- Taus link 2 ---\n";
    unpackTaus(bl->tau2, bx, 6);

    // unpack et sums
    LogDebug("L1Scout") << "--- Sums ---\n";
    unpackEtSums(bl->sum, bx);

  }  // end of bx objects
}

void ScCaloRawToDigi::unpackTcpData(edm::Handle<SDSRawDataCollection>& ScoutingRawDataCollection,
                                    std::vector<int> sourceList,
                                    CaloObjectType dataType) {
  for (const int& sourceId : sourceList) {
    if ((sourceId < SDSNumbering::CaloTCPMinSDSID) || (sourceId > SDSNumbering::CaloTCPMaxSDSID)) {
      edm::LogWarning("ScCaloRawToDIgi::unpackTCPData")
          << "Source ID outside the expected range " << sourceId << "/[" << SDSNumbering::CaloTCPMinSDSID << "-"
          << SDSNumbering::CaloTCPMaxSDSID << "]"
          << " [provided/expected range]";
    }
    const FEDRawData& sourceRawData = ScoutingRawDataCollection->FEDData(sourceId);
    size_t orbitSize = sourceRawData.size();

    if (sourceRawData.size() == 0) {
      LogDebug("L1Scout") << "No raw data for calo TCP source ID=" << sourceId << "\n";
    }

    unpackOrbitFromTCP(sourceRawData.data(), orbitSize, dataType);
  }
}

void ScCaloRawToDigi::unpackOrbitFromTCP(const unsigned char* buf, size_t len, CaloObjectType dataType) {
  size_t pos = 0;

  while (pos < len) {
    // frame header is present
    assert(pos + 4 <= len);

    // unpack calo sums block
    if (dataType == CaloObjectType::EtSum) {
      l1ScoutingRun3::demux::caloSumTcpBlock* bl = (l1ScoutingRun3::demux::caloSumTcpBlock*)(buf + pos);
      pos += sizeof(l1ScoutingRun3::demux::caloSumTcpBlock);
      assert(pos <= len);
      LogDebug("L1Scout") << "Sums BX -> " << bl->bx;
      unpackEtSums(bl->sum, bl->bx);
    } else {
      // unpack jet/eg/tau
      l1ScoutingRun3::demux::caloObjTcpBlock* bl = (l1ScoutingRun3::demux::caloObjTcpBlock*)(buf + pos);
      int nObj = (bl->header) & 0xff;
      pos += 12 + nObj * 4;

      switch (dataType) {
        case CaloObjectType::Jet:
          LogDebug("L1Scout") << "Jets BX -> " << bl->bx;
          unpackJets(bl->obj, bl->bx, nObj);
          break;

        case CaloObjectType::EGamma:
          LogDebug("L1Scout") << "E/Gammas BX -> " << bl->bx;
          unpackEGammas(bl->obj, bl->bx, nObj);
          break;

        case CaloObjectType::Tau:
          LogDebug("L1Scout") << "Taus BX -> " << bl->bx;
          unpackTaus(bl->obj, bl->bx, nObj);
          break;

        default:
          throw cms::Exception("ScCaloRawToDigi::unpackOrbitFromTCP") << "Unknown data type.";
          break;
      }

    }  // unpack sums and calo objects

  }  // end of bx objects
}

void ScCaloRawToDigi::unpackJets(uint32_t* dataBlock, int bx, int nObjets) {
  int32_t ET(0), Eta(0), Phi(0), Qual(0);
  for (int i = 0; i < nObjets; i++) {
    ET = ((dataBlock[i] >> l1ScoutingRun3::demux::shiftsJet::ET) & l1ScoutingRun3::demux::masksJet::ET);

    if (ET != 0) {
      Eta = ((dataBlock[i] >> l1ScoutingRun3::demux::shiftsJet::eta) & l1ScoutingRun3::demux::masksJet::eta);
      Phi = ((dataBlock[i] >> l1ScoutingRun3::demux::shiftsJet::phi) & l1ScoutingRun3::demux::masksJet::phi);
      Qual = ((dataBlock[i] >> l1ScoutingRun3::demux::shiftsJet::qual) & l1ScoutingRun3::demux::masksJet::qual);

      if (Eta > 127)
        Eta = Eta - 256;

      l1ScoutingRun3::Jet jet(ET, Eta, Phi, Qual);
      orbitBufferJets_[bx].push_back(jet);
      nJetsOrbit_++;
      if (edm::MessageDrop::instance()->debugEnabled) {
        std::ostringstream os;
        os << "Jet " << i << "\n";
        os << "  Raw: 0x" << std::hex << dataBlock[i] << std::dec << "\n";
        l1ScoutingRun3::printJet(jet, os);
        LogDebug("L1Scout") << os.str();
      }
    }
  }  // end link jets unpacking loop
}

void ScCaloRawToDigi::unpackEGammas(uint32_t* dataBlock, int bx, int nObjets) {
  int32_t ET(0), Eta(0), Phi(0), Iso(0);
  for (int i = 0; i < nObjets; i++) {
    ET = ((dataBlock[i] >> l1ScoutingRun3::demux::shiftsEGamma::ET) & l1ScoutingRun3::demux::masksEGamma::ET);
    if (ET != 0) {
      Eta = ((dataBlock[i] >> l1ScoutingRun3::demux::shiftsEGamma::eta) & l1ScoutingRun3::demux::masksEGamma::eta);
      Phi = ((dataBlock[i] >> l1ScoutingRun3::demux::shiftsEGamma::phi) & l1ScoutingRun3::demux::masksEGamma::phi);
      Iso = ((dataBlock[i] >> l1ScoutingRun3::demux::shiftsEGamma::iso) & l1ScoutingRun3::demux::masksEGamma::iso);

      if (Eta > 127)
        Eta = Eta - 256;

      l1ScoutingRun3::EGamma eGamma(ET, Eta, Phi, Iso);
      orbitBufferEGammas_[bx].push_back(eGamma);
      nEGammasOrbit_++;

      if (edm::MessageDrop::instance()->debugEnabled) {
        std::ostringstream os;
        os << "E/g " << i << "\n";
        os << "  Raw: 0x" << std::hex << dataBlock[i] << std::dec << "\n";
        l1ScoutingRun3::printEGamma(eGamma, os);
        LogDebug("L1Scout") << os.str();
      }
    }
  }  // end link e/gammas unpacking loop
}

void ScCaloRawToDigi::unpackTaus(uint32_t* dataBlock, int bx, int nObjets) {
  int32_t ET(0), Eta(0), Phi(0), Iso(0);
  for (int i = 0; i < nObjets; i++) {
    ET = ((dataBlock[i] >> l1ScoutingRun3::demux::shiftsTau::ET) & l1ScoutingRun3::demux::masksTau::ET);
    if (ET != 0) {
      Eta = ((dataBlock[i] >> l1ScoutingRun3::demux::shiftsTau::eta) & l1ScoutingRun3::demux::masksTau::eta);
      Phi = ((dataBlock[i] >> l1ScoutingRun3::demux::shiftsTau::phi) & l1ScoutingRun3::demux::masksTau::phi);
      Iso = ((dataBlock[i] >> l1ScoutingRun3::demux::shiftsTau::iso) & l1ScoutingRun3::demux::masksTau::iso);

      if (Eta > 127)
        Eta = Eta - 256;

      l1ScoutingRun3::Tau tau(ET, Eta, Phi, Iso);
      orbitBufferTaus_[bx].push_back(tau);
      nTausOrbit_++;

      if (edm::MessageDrop::instance()->debugEnabled) {
        std::ostringstream os;
        os << "Tau " << i << "\n";
        os << "  Raw: 0x" << std::hex << dataBlock[i] << std::dec << "\n";
        l1ScoutingRun3::printTau(tau, os);
        LogDebug("L1Scout") << os.str();
      }
    }
  }  // end link taus unpacking loop
}

void ScCaloRawToDigi::unpackEtSums(uint32_t* dataBlock, int bx) {
  l1ScoutingRun3::BxSums bxSums;

  int32_t ETEt(0), ETEttem(0), ETMinBiasHFP0(0);                                // ET block
  int32_t HTEt(0), HTtowerCount(0), HTMinBiasHFM0(0);                           // HT block
  int32_t ETmissEt(0), ETmissPhi(0), ETmissASYMET(0), ETmissMinBiasHFP1(0);     // ETMiss block
  int32_t HTmissEt(0), HTmissPhi(0), HTmissASYMHT(0), HTmissMinBiasHFM1(0);     // HTMiss block
  int32_t ETHFmissEt(0), ETHFmissPhi(0), ETHFmissASYMETHF(0), ETHFmissCENT(0);  // ETHFMiss block
  int32_t HTHFmissEt(0), HTHFmissPhi(0), HTHFmissASYMHTHF(0), HTHFmissCENT(0);  // HTHFMiss block

  // ET block
  ETEt = ((dataBlock[0] >> l1ScoutingRun3::demux::shiftsESums::ETEt) & l1ScoutingRun3::demux::masksESums::ETEt);
  ETEttem =
      ((dataBlock[0] >> l1ScoutingRun3::demux::shiftsESums::ETEttem) & l1ScoutingRun3::demux::masksESums::ETEttem);

  bxSums.setHwTotalEt(ETEt);
  bxSums.setHwTotalEtEm(ETEttem);

  // HT block
  HTEt = ((dataBlock[1] >> l1ScoutingRun3::demux::shiftsESums::HTEt) & l1ScoutingRun3::demux::masksESums::HTEt);

  bxSums.setHwTotalHt(HTEt);

  // ETMiss block
  ETmissEt =
      ((dataBlock[2] >> l1ScoutingRun3::demux::shiftsESums::ETmissEt) & l1ScoutingRun3::demux::masksESums::ETmissEt);
  ETmissPhi =
      ((dataBlock[2] >> l1ScoutingRun3::demux::shiftsESums::ETmissPhi) & l1ScoutingRun3::demux::masksESums::ETmissPhi);

  if (ETmissEt > 0) {
    bxSums.setHwMissEt(ETmissEt);
    bxSums.setHwMissEtPhi(ETmissPhi);
  }

  // HTMiss block
  HTmissEt =
      ((dataBlock[3] >> l1ScoutingRun3::demux::shiftsESums::HTmissEt) & l1ScoutingRun3::demux::masksESums::HTmissEt);
  HTmissPhi =
      ((dataBlock[3] >> l1ScoutingRun3::demux::shiftsESums::HTmissPhi) & l1ScoutingRun3::demux::masksESums::HTmissPhi);

  if (HTmissEt > 0) {
    bxSums.setHwMissHt(HTmissEt);
    bxSums.setHwMissHtPhi(HTmissPhi);
  }

  // ETHFMiss block
  ETHFmissEt = ((dataBlock[4] >> l1ScoutingRun3::demux::shiftsESums::ETHFmissEt) &
                l1ScoutingRun3::demux::masksESums::ETHFmissEt);
  ETHFmissPhi = ((dataBlock[4] >> l1ScoutingRun3::demux::shiftsESums::ETHFmissPhi) &
                 l1ScoutingRun3::demux::masksESums::ETHFmissPhi);

  if (ETHFmissEt > 0) {
    bxSums.setHwMissEtHF(ETHFmissEt);
    bxSums.setHwMissEtHFPhi(ETHFmissPhi);
  }

  // HTHFMiss block
  HTHFmissEt = ((dataBlock[5] >> l1ScoutingRun3::demux::shiftsESums::ETHFmissEt) &
                l1ScoutingRun3::demux::masksESums::ETHFmissEt);
  HTHFmissPhi = ((dataBlock[5] >> l1ScoutingRun3::demux::shiftsESums::ETHFmissPhi) &
                 l1ScoutingRun3::demux::masksESums::ETHFmissPhi);

  if (HTHFmissEt > 0) {
    bxSums.setHwMissHtHF(HTHFmissEt);
    bxSums.setHwMissHtHFPhi(HTHFmissPhi);
  }

  // Insert additional sums
  if (enableAllSums_) {
    // ET block
    ETMinBiasHFP0 = ((dataBlock[0] >> l1ScoutingRun3::demux::shiftsESums::ETMinBiasHF) &
                     l1ScoutingRun3::demux::masksESums::ETMinBiasHF);
    bxSums.setMinBiasHFP0(ETMinBiasHFP0);

    // HT block
    HTtowerCount = ((dataBlock[1] >> l1ScoutingRun3::demux::shiftsESums::HTtowerCount) &
                    l1ScoutingRun3::demux::masksESums::HTtowerCount);
    HTMinBiasHFM0 = ((dataBlock[1] >> l1ScoutingRun3::demux::shiftsESums::HTMinBiasHF) &
                     l1ScoutingRun3::demux::masksESums::HTMinBiasHF);

    bxSums.setTowerCount(HTtowerCount);
    bxSums.setMinBiasHFM0(HTMinBiasHFM0);

    // ET Miss block
    ETmissASYMET = ((dataBlock[2] >> l1ScoutingRun3::demux::shiftsESums::ETmissASYMET) &
                    l1ScoutingRun3::demux::masksESums::ETmissASYMET);
    ETmissMinBiasHFP1 = ((dataBlock[2] >> l1ScoutingRun3::demux::shiftsESums::ETmissMinBiasHF) &
                         l1ScoutingRun3::demux::masksESums::ETmissMinBiasHF);
    bxSums.setHwAsymEt(ETmissASYMET);
    bxSums.setMinBiasHFP1(ETmissMinBiasHFP1);

    // HT Miss block
    HTmissASYMHT = ((dataBlock[3] >> l1ScoutingRun3::demux::shiftsESums::HTmissASYMHT) &
                    l1ScoutingRun3::demux::masksESums::HTmissASYMHT);
    HTmissMinBiasHFM1 = ((dataBlock[3] >> l1ScoutingRun3::demux::shiftsESums::HTmissMinBiasHF) &
                         l1ScoutingRun3::demux::masksESums::HTmissMinBiasHF);

    bxSums.setHwAsymHt(HTmissASYMHT);
    bxSums.setMinBiasHFM1(HTmissMinBiasHFM1);

    // ETHFMiss
    ETHFmissASYMETHF = ((dataBlock[4] >> l1ScoutingRun3::demux::shiftsESums::ETHFmissASYMETHF) &
                        l1ScoutingRun3::demux::masksESums::ETHFmissASYMETHF);
    ETHFmissCENT = ((dataBlock[4] >> l1ScoutingRun3::demux::shiftsESums::ETHFmissCENT) &
                    l1ScoutingRun3::demux::masksESums::ETHFmissCENT);

    bxSums.setHwAsymEtHF(ETHFmissASYMETHF);

    // HTHFMiss
    HTHFmissASYMHTHF = ((dataBlock[5] >> l1ScoutingRun3::demux::shiftsESums::ETHFmissASYMETHF) &
                        l1ScoutingRun3::demux::masksESums::ETHFmissASYMETHF);
    HTHFmissCENT = ((dataBlock[5] >> l1ScoutingRun3::demux::shiftsESums::ETHFmissCENT) &
                    l1ScoutingRun3::demux::masksESums::ETHFmissCENT);

    bxSums.setHwAsymHtHF(HTHFmissASYMHTHF);
    bxSums.setCentrality((HTHFmissCENT << 4) + ETHFmissCENT);
  }

  orbitBufferEtSums_[bx].push_back(bxSums);
  nEtSumsOrbit_ += 1;

  if (edm::MessageDrop::instance()->debugEnabled) {
    std::ostringstream os;
    os << "Raw frames:\n";
    for (int frame = 0; frame < 6; frame++) {
      os << "  frame " << frame << ": 0x" << std::hex << dataBlock[frame] << std::dec << "\n";
      l1ScoutingRun3::printBxSums(bxSums, os);
    }
    LogDebug("L1Scout") << os.str();
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
    dataSource.add<int>("dmaSourceId", 2);
    desc.add("dataSource", dataSource);
  }
  desc.add<bool>("enableAllSums", true);
  desc.addUntracked<bool>("debug", false);
  descriptions.add("ScCaloRawToDigi", desc);
}

DEFINE_FWK_MODULE(ScCaloRawToDigi);
