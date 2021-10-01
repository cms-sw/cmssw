#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"

#include "EcalTPGDBCopy.h"

#include <vector>

EcalTPGDBCopy::EcalTPGDBCopy(const edm::ParameterSet& iConfig)
    : m_timetype(iConfig.getParameter<std::string>("timetype")), m_cacheIDs(), m_records() {
  auto cc = consumesCollector();
  std::string container;
  std::string tag;
  std::string record;
  typedef std::vector<edm::ParameterSet> Parameters;
  Parameters toCopy = iConfig.getParameter<Parameters>("toCopy");
  for (Parameters::iterator i = toCopy.begin(); i != toCopy.end(); ++i) {
    container = i->getParameter<std::string>("container");
    record = i->getParameter<std::string>("record");
    m_cacheIDs.insert(std::make_pair(container, 0));
    m_records.insert(std::make_pair(container, record));
    setConsumes(cc, container);
  }
}

EcalTPGDBCopy::~EcalTPGDBCopy() {}

void EcalTPGDBCopy::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) {
  std::string container;
  std::string record;
  typedef std::map<std::string, std::string>::const_iterator recordIter;
  for (recordIter i = m_records.begin(); i != m_records.end(); ++i) {
    container = (*i).first;
    record = (*i).second;
    if (shouldCopy(evtSetup, container)) {
      copyToDB(evtSetup, container);
    }
  }
}

void EcalTPGDBCopy::setConsumes(edm::ConsumesCollector& cc, const std::string& container) {
  if (container == "EcalTPGPedestals") {
    pedestalsToken_ = cc.esConsumes();
  } else if (container == "EcalTPGLinearizationConst") {
    linearizationConstToken_ = cc.esConsumes();
  } else if (container == "EcalTPGSlidingWindow") {
    slidingWindowToken_ = cc.esConsumes();
  } else if (container == "EcalTPGFineGrainEBIdMap") {
    fineGrainEBIdMapToken_ = cc.esConsumes();
  } else if (container == "EcalTPGFineGrainStripEE") {
    fineGrainStripEEToken_ = cc.esConsumes();
  } else if (container == "EcalTPGFineGrainTowerEE") {
    fineGrainTowerEEToken_ = cc.esConsumes();
  } else if (container == "EcalTPGLutIdMap") {
    lutIdMapToken_ = cc.esConsumes();
  } else if (container == "EcalTPGWeightIdMap") {
    weightIdMapToken_ = cc.esConsumes();
  } else if (container == "EcalTPGWeightGroup") {
    weightGroupToken_ = cc.esConsumes();
  } else if (container == "EcalTPGOddWeightIdMap") {
    oddWeightIdMapToken_ = cc.esConsumes();
  } else if (container == "EcalTPGOddWeightGroup") {
    oddWeightGroupToken_ = cc.esConsumes();
  } else if (container == "EcalTPGTPMode") {
    tpModeToken_ = cc.esConsumes();
  } else if (container == "EcalTPGLutGroup") {
    lutGroupToken_ = cc.esConsumes();
  } else if (container == "EcalTPGFineGrainEBGroup") {
    fineGrainEBGroupToken_ = cc.esConsumes();
  } else if (container == "EcalTPGPhysicsConst") {
    physicsConstToken_ = cc.esConsumes();
  } else if (container == "EcalTPGCrystalStatus") {
    crystalStatusToken_ = cc.esConsumes();
  } else if (container == "EcalTPGTowerStatus") {
    towerStatusToken_ = cc.esConsumes();
  } else if (container == "EcalTPGSpike") {
    spikeToken_ = cc.esConsumes();
  } else if (container == "EcalTPGStripStatus") {
    stripStatusToken_ = cc.esConsumes();
  } else {
    throw cms::Exception("Unknown container");
  }
}

bool EcalTPGDBCopy::shouldCopy(const edm::EventSetup& evtSetup, const std::string& container) {
  unsigned long long cacheID = 0;

  if (container == "EcalTPGPedestals") {
    cacheID = evtSetup.get<EcalTPGPedestalsRcd>().cacheIdentifier();
  } else if (container == "EcalTPGLinearizationConst") {
    cacheID = evtSetup.get<EcalTPGLinearizationConstRcd>().cacheIdentifier();
  } else if (container == "EcalTPGSlidingWindow") {
    cacheID = evtSetup.get<EcalTPGSlidingWindowRcd>().cacheIdentifier();
  } else if (container == "EcalTPGFineGrainEBIdMap") {
    cacheID = evtSetup.get<EcalTPGFineGrainEBIdMapRcd>().cacheIdentifier();
  } else if (container == "EcalTPGFineGrainStripEE") {
    cacheID = evtSetup.get<EcalTPGFineGrainStripEERcd>().cacheIdentifier();
  } else if (container == "EcalTPGFineGrainTowerEE") {
    cacheID = evtSetup.get<EcalTPGFineGrainTowerEERcd>().cacheIdentifier();
  } else if (container == "EcalTPGLutIdMap") {
    cacheID = evtSetup.get<EcalTPGLutIdMapRcd>().cacheIdentifier();
  } else if (container == "EcalTPGWeightIdMap") {
    cacheID = evtSetup.get<EcalTPGWeightIdMapRcd>().cacheIdentifier();
  } else if (container == "EcalTPGWeightGroup") {
    cacheID = evtSetup.get<EcalTPGWeightGroupRcd>().cacheIdentifier();
  } else if (container == "EcalTPGOddWeightIdMap") {
    cacheID = evtSetup.get<EcalTPGOddWeightIdMapRcd>().cacheIdentifier();
  } else if (container == "EcalTPGOddWeightGroup") {
    cacheID = evtSetup.get<EcalTPGOddWeightGroupRcd>().cacheIdentifier();
  } else if (container == "EcalTPGTPMode") {
    cacheID = evtSetup.get<EcalTPGTPModeRcd>().cacheIdentifier();
  } else if (container == "EcalTPGLutGroup") {
    cacheID = evtSetup.get<EcalTPGLutGroupRcd>().cacheIdentifier();
  } else if (container == "EcalTPGFineGrainEBGroup") {
    cacheID = evtSetup.get<EcalTPGFineGrainEBGroupRcd>().cacheIdentifier();
  } else if (container == "EcalTPGPhysicsConst") {
    cacheID = evtSetup.get<EcalTPGPhysicsConstRcd>().cacheIdentifier();
  } else if (container == "EcalTPGCrystalStatus") {
    cacheID = evtSetup.get<EcalTPGCrystalStatusRcd>().cacheIdentifier();
  } else if (container == "EcalTPGTowerStatus") {
    cacheID = evtSetup.get<EcalTPGTowerStatusRcd>().cacheIdentifier();
  } else if (container == "EcalTPGSpike") {
    cacheID = evtSetup.get<EcalTPGSpikeRcd>().cacheIdentifier();
  } else if (container == "EcalTPGStripStatus") {
    cacheID = evtSetup.get<EcalTPGStripStatusRcd>().cacheIdentifier();
  } else {
    throw cms::Exception("Unknown container");
  }

  if (m_cacheIDs[container] == cacheID) {
    return false;
  } else {
    m_cacheIDs[container] = cacheID;
    return true;
  }
}

void EcalTPGDBCopy::copyToDB(const edm::EventSetup& evtSetup, const std::string& container) {
  edm::Service<cond::service::PoolDBOutputService> dbOutput;
  if (!dbOutput.isAvailable()) {
    throw cms::Exception("PoolDBOutputService is not available");
  }

  std::string recordName = m_records[container];

  if (container == "EcalTPGPedestals") {
    const auto handle = evtSetup.getHandle(pedestalsToken_);
    const EcalTPGPedestals* obj = handle.product();
    dbOutput->createNewIOV<const EcalTPGPedestals>(
        new EcalTPGPedestals(*obj), dbOutput->beginOfTime(), dbOutput->endOfTime(), recordName);

  } else if (container == "EcalTPGLinearizationConst") {
    const auto handle = evtSetup.getHandle(linearizationConstToken_);
    const EcalTPGLinearizationConst* obj = handle.product();
    dbOutput->createNewIOV<const EcalTPGLinearizationConst>(
        new EcalTPGLinearizationConst(*obj), dbOutput->beginOfTime(), dbOutput->endOfTime(), recordName);

  } else if (container == "EcalTPGSlidingWindow") {
    const auto handle = evtSetup.getHandle(slidingWindowToken_);
    const EcalTPGSlidingWindow* obj = handle.product();
    dbOutput->createNewIOV<const EcalTPGSlidingWindow>(
        new EcalTPGSlidingWindow(*obj), dbOutput->beginOfTime(), dbOutput->endOfTime(), recordName);

  } else if (container == "EcalTPGFineGrainEBIdMap") {
    const auto handle = evtSetup.getHandle(fineGrainEBIdMapToken_);
    const EcalTPGFineGrainEBIdMap* obj = handle.product();
    dbOutput->createNewIOV<const EcalTPGFineGrainEBIdMap>(
        new EcalTPGFineGrainEBIdMap(*obj), dbOutput->beginOfTime(), dbOutput->endOfTime(), recordName);

  } else if (container == "EcalTPGFineGrainStripEE") {
    const auto handle = evtSetup.getHandle(fineGrainStripEEToken_);
    const EcalTPGFineGrainStripEE* obj = handle.product();
    dbOutput->createNewIOV<const EcalTPGFineGrainStripEE>(
        new EcalTPGFineGrainStripEE(*obj), dbOutput->beginOfTime(), dbOutput->endOfTime(), recordName);

  } else if (container == "EcalTPGFineGrainTowerEE") {
    const auto handle = evtSetup.getHandle(fineGrainTowerEEToken_);
    const EcalTPGFineGrainTowerEE* obj = handle.product();
    dbOutput->createNewIOV<const EcalTPGFineGrainTowerEE>(
        new EcalTPGFineGrainTowerEE(*obj), dbOutput->beginOfTime(), dbOutput->endOfTime(), recordName);

  } else if (container == "EcalTPGLutIdMap") {
    const auto handle = evtSetup.getHandle(lutIdMapToken_);
    const EcalTPGLutIdMap* obj = handle.product();
    dbOutput->createNewIOV<const EcalTPGLutIdMap>(
        new EcalTPGLutIdMap(*obj), dbOutput->beginOfTime(), dbOutput->endOfTime(), recordName);

  } else if (container == "EcalTPGWeightIdMap") {
    const auto handle = evtSetup.getHandle(weightIdMapToken_);
    const EcalTPGWeightIdMap* obj = handle.product();
    dbOutput->createNewIOV<const EcalTPGWeightIdMap>(
        new EcalTPGWeightIdMap(*obj), dbOutput->beginOfTime(), dbOutput->endOfTime(), recordName);

  } else if (container == "EcalTPGWeightGroup") {
    const auto handle = evtSetup.getHandle(weightGroupToken_);
    const EcalTPGWeightGroup* obj = handle.product();
    dbOutput->createNewIOV<const EcalTPGWeightGroup>(
        new EcalTPGWeightGroup(*obj), dbOutput->beginOfTime(), dbOutput->endOfTime(), recordName);

  } else if (container == "EcalTPGOddWeightIdMap") {
    const auto handle = evtSetup.getHandle(oddWeightIdMapToken_);
    const EcalTPGOddWeightIdMap* obj = handle.product();
    dbOutput->createNewIOV<const EcalTPGOddWeightIdMap>(
        new EcalTPGOddWeightIdMap(*obj), dbOutput->beginOfTime(), dbOutput->endOfTime(), recordName);

  } else if (container == "EcalTPGOddWeightGroup") {
    const auto handle = evtSetup.getHandle(oddWeightGroupToken_);
    const EcalTPGOddWeightGroup* obj = handle.product();
    dbOutput->createNewIOV<const EcalTPGOddWeightGroup>(
        new EcalTPGOddWeightGroup(*obj), dbOutput->beginOfTime(), dbOutput->endOfTime(), recordName);

  } else if (container == "EcalTPGTPMode") {
    const auto handle = evtSetup.getHandle(tpModeToken_);
    const EcalTPGTPMode* obj = handle.product();
    dbOutput->createNewIOV<const EcalTPGTPMode>(
        new EcalTPGTPMode(*obj), dbOutput->beginOfTime(), dbOutput->endOfTime(), recordName);

  } else if (container == "EcalTPGLutGroup") {
    const auto handle = evtSetup.getHandle(lutGroupToken_);
    const EcalTPGLutGroup* obj = handle.product();
    dbOutput->createNewIOV<const EcalTPGLutGroup>(
        new EcalTPGLutGroup(*obj), dbOutput->beginOfTime(), dbOutput->endOfTime(), recordName);

  } else if (container == "EcalTPGFineGrainEBGroup") {
    const auto handle = evtSetup.getHandle(fineGrainEBGroupToken_);
    const EcalTPGFineGrainEBGroup* obj = handle.product();
    dbOutput->createNewIOV<const EcalTPGFineGrainEBGroup>(
        new EcalTPGFineGrainEBGroup(*obj), dbOutput->beginOfTime(), dbOutput->endOfTime(), recordName);

  } else if (container == "EcalTPGPhysicsConst") {
    const auto handle = evtSetup.getHandle(physicsConstToken_);
    const EcalTPGPhysicsConst* obj = handle.product();
    dbOutput->createNewIOV<const EcalTPGPhysicsConst>(
        new EcalTPGPhysicsConst(*obj), dbOutput->beginOfTime(), dbOutput->endOfTime(), recordName);

  } else if (container == "EcalTPGCrystalStatus") {
    const auto handle = evtSetup.getHandle(crystalStatusToken_);
    const EcalTPGCrystalStatus* obj = handle.product();
    dbOutput->createNewIOV<const EcalTPGCrystalStatus>(
        new EcalTPGCrystalStatus(*obj), dbOutput->beginOfTime(), dbOutput->endOfTime(), recordName);

  } else if (container == "EcalTPGTowerStatus") {
    const auto handle = evtSetup.getHandle(towerStatusToken_);
    const EcalTPGTowerStatus* obj = handle.product();
    dbOutput->createNewIOV<const EcalTPGTowerStatus>(
        new EcalTPGTowerStatus(*obj), dbOutput->beginOfTime(), dbOutput->endOfTime(), recordName);

  } else if (container == "EcalTPGSpike") {
    const auto handle = evtSetup.getHandle(spikeToken_);
    const EcalTPGSpike* obj = handle.product();

    dbOutput->createNewIOV<const EcalTPGSpike>(
        new EcalTPGSpike(*obj), dbOutput->beginOfTime(), dbOutput->endOfTime(), recordName);

  } else if (container == "EcalTPGStripStatus") {
    const auto handle = evtSetup.getHandle(stripStatusToken_);
    const EcalTPGStripStatus* obj = handle.product();
    dbOutput->createNewIOV<const EcalTPGStripStatus>(
        new EcalTPGStripStatus(*obj), dbOutput->beginOfTime(), dbOutput->endOfTime(), recordName);

  } else {
    throw cms::Exception("Unknown container");
  }

  edm::LogInfo("EcalTPGDBCopy") << "EcalTPGDBCopy wrote " << recordName;
}
