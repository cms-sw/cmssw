#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"

#include "CondTools/Ecal/interface/ESDBCopy.h"
#include <vector>

ESDBCopy::ESDBCopy(const edm::ParameterSet& iConfig)
    : m_timetype(iConfig.getParameter<std::string>("timetype")),
      m_cacheIDs(),
      m_records(),
      esPedestalsToken_(esConsumes()),
      esADCToGeVConstantToken_(esConsumes()),
      esChannelStatusToken_(esConsumes()),
      esIntercalibConstantsToken_(esConsumes()),
      esWeightStripGroupsToken_(esConsumes()),
      esTBWeightsToken_(esConsumes()) {
  std::string container;
  std::string record;
  typedef std::vector<edm::ParameterSet> Parameters;
  Parameters toCopy = iConfig.getParameter<Parameters>("toCopy");
  for (const auto& iparam : toCopy) {
    container = iparam.getParameter<std::string>("container");
    record = iparam.getParameter<std::string>("record");
    m_cacheIDs.emplace(container, 0);
    m_records.emplace(container, record);
  }
}

ESDBCopy::~ESDBCopy() {}

void ESDBCopy::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) {
  for (const auto& irec : m_records) {
    if (shouldCopy(evtSetup, irec.first)) {
      copyToDB(evtSetup, irec.first);
    }
  }
}

bool ESDBCopy::shouldCopy(const edm::EventSetup& evtSetup, const std::string& container) {
  unsigned long long cacheID = 0;
  if (container == "ESPedestals") {
    cacheID = evtSetup.get<ESPedestalsRcd>().cacheIdentifier();
  } else if (container == "ESADCToGeVConstant") {
    cacheID = evtSetup.get<ESADCToGeVConstantRcd>().cacheIdentifier();
  } else if (container == "ESIntercalibConstants") {
    cacheID = evtSetup.get<ESIntercalibConstantsRcd>().cacheIdentifier();
  } else if (container == "ESWeightStripGroups") {
    cacheID = evtSetup.get<ESWeightStripGroupsRcd>().cacheIdentifier();
  } else if (container == "ESTBWeights") {
    cacheID = evtSetup.get<ESTBWeightsRcd>().cacheIdentifier();
  } else if (container == "ESChannelStatus") {
    cacheID = evtSetup.get<ESChannelStatusRcd>().cacheIdentifier();
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

void ESDBCopy::copyToDB(const edm::EventSetup& evtSetup, const std::string& container) {
  edm::Service<cond::service::PoolDBOutputService> dbOutput;
  if (!dbOutput.isAvailable()) {
    throw cms::Exception("PoolDBOutputService is not available");
  }

  std::string recordName = m_records[container];

  if (container == "ESPedestals") {
    const auto& obj = evtSetup.getData(esPedestalsToken_);
    edm::LogInfo("ESDBCopy") << "ped pointer is: " << &obj;
    dbOutput->createOneIOV<const ESPedestals>(obj, dbOutput->beginOfTime(), recordName);

  } else if (container == "ESADCToGeVConstant") {
    const auto& obj = evtSetup.getData(esADCToGeVConstantToken_);
    edm::LogInfo("ESDBCopy") << "adc pointer is: " << &obj;
    dbOutput->createOneIOV<const ESADCToGeVConstant>(obj, dbOutput->beginOfTime(), recordName);

  } else if (container == "ESChannelStatus") {
    const auto& obj = evtSetup.getData(esChannelStatusToken_);
    edm::LogInfo("ESDBCopy") << "channel status pointer is: " << &obj;
    dbOutput->createOneIOV<const ESChannelStatus>(obj, dbOutput->beginOfTime(), recordName);

  } else if (container == "ESIntercalibConstants") {
    const auto& obj = evtSetup.getData(esIntercalibConstantsToken_);
    edm::LogInfo("ESDBCopy") << "inter pointer is: " << &obj;
    dbOutput->createOneIOV<const ESIntercalibConstants>(obj, dbOutput->beginOfTime(), recordName);

  } else if (container == "ESWeightStripGroups") {
    const auto& obj = evtSetup.getData(esWeightStripGroupsToken_);
    edm::LogInfo("ESDBCopy") << "weight pointer is: " << &obj;
    dbOutput->createOneIOV<const ESWeightStripGroups>(obj, dbOutput->beginOfTime(), recordName);

  } else if (container == "ESTBWeights") {
    const auto& obj = evtSetup.getData(esTBWeightsToken_);
    edm::LogInfo("ESDBCopy") << "tbweight pointer is: " << &obj;
    dbOutput->createOneIOV<const ESTBWeights>(obj, dbOutput->beginOfTime(), recordName);

  } else {
    throw cms::Exception("Unknown container");
  }

  edm::LogInfo("ESDBCopy") << "ESDBCopy wrote " << recordName;
}
