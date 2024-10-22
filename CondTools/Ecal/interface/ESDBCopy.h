#ifndef CondTools_Ecal_ESDBCopy_h
#define CondTools_Ecal_ESDBCopy_h

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "CondCore/CondDB/interface/Exception.h"

#include "CondFormats/ESObjects/interface/ESPedestals.h"
#include "CondFormats/DataRecord/interface/ESPedestalsRcd.h"
#include "CondFormats/ESObjects/interface/ESADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/ESADCToGeVConstantRcd.h"
#include "CondFormats/ESObjects/interface/ESChannelStatus.h"
#include "CondFormats/DataRecord/interface/ESChannelStatusRcd.h"
#include "CondFormats/ESObjects/interface/ESIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/ESIntercalibConstantsRcd.h"
#include "CondFormats/ESObjects/interface/ESWeightStripGroups.h"
#include "CondFormats/DataRecord/interface/ESWeightStripGroupsRcd.h"
#include "CondFormats/ESObjects/interface/ESTBWeights.h"
#include "CondFormats/DataRecord/interface/ESTBWeightsRcd.h"

#include "FWCore/Framework/interface/IOVSyncValue.h"

#include <string>
#include <map>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

class ESDBCopy : public edm::one::EDAnalyzer<> {
public:
  explicit ESDBCopy(const edm::ParameterSet& iConfig);
  ~ESDBCopy() override;

  void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) override;

private:
  bool shouldCopy(const edm::EventSetup& evtSetup, const std::string& container);
  void copyToDB(const edm::EventSetup& evtSetup, const std::string& container);

  std::string m_timetype;
  std::map<std::string, unsigned long long> m_cacheIDs;
  std::map<std::string, std::string> m_records;

  const edm::ESGetToken<ESPedestals, ESPedestalsRcd> esPedestalsToken_;
  const edm::ESGetToken<ESADCToGeVConstant, ESADCToGeVConstantRcd> esADCToGeVConstantToken_;
  const edm::ESGetToken<ESChannelStatus, ESChannelStatusRcd> esChannelStatusToken_;
  const edm::ESGetToken<ESIntercalibConstants, ESIntercalibConstantsRcd> esIntercalibConstantsToken_;
  const edm::ESGetToken<ESWeightStripGroups, ESWeightStripGroupsRcd> esWeightStripGroupsToken_;
  const edm::ESGetToken<ESTBWeights, ESTBWeightsRcd> esTBWeightsToken_;
};

#endif
