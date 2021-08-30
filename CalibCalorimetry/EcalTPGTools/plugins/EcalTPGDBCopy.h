#ifndef CalibCalorimetry_EcalTPGTools_EcalTPGDBCopy_h
#define CalibCalorimetry_EcalTPGTools_EcalTPGDBCopy_h

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "CondCore/CondDB/interface/Exception.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"

#include "CondFormats/EcalObjects/interface/EcalTPGPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalTPGLinearizationConst.h"
#include "CondFormats/EcalObjects/interface/EcalTPGSlidingWindow.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainEBIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainStripEE.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainTowerEE.h"
#include "CondFormats/EcalObjects/interface/EcalTPGLutIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalTPGWeightIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalTPGWeightGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGOddWeightIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalTPGOddWeightGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGTPMode.h"
#include "CondFormats/EcalObjects/interface/EcalTPGLutGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainEBGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGPhysicsConst.h"
#include "CondFormats/EcalObjects/interface/EcalTPGCrystalStatus.h"
#include "CondFormats/EcalObjects/interface/EcalTPGSpike.h"
#include "CondFormats/EcalObjects/interface/EcalTPGTowerStatus.h"
#include "CondFormats/EcalObjects/interface/EcalTPGStripStatus.h"

#include "CondFormats/DataRecord/interface/EcalTPGPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGLinearizationConstRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGSlidingWindowRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGFineGrainEBIdMapRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGFineGrainStripEERcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGFineGrainTowerEERcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGLutIdMapRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGWeightIdMapRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGWeightGroupRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGOddWeightIdMapRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGOddWeightGroupRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGTPModeRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGLutGroupRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGFineGrainEBGroupRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGPhysicsConstRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGCrystalStatusRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGTowerStatusRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGSpikeRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGStripStatusRcd.h"

#include <string>
#include <map>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

class EcalTPGDBCopy : public edm::one::EDAnalyzer<> {
public:
  explicit EcalTPGDBCopy(const edm::ParameterSet& iConfig);
  ~EcalTPGDBCopy() override;

  void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) override;

private:
  void setConsumes(edm::ConsumesCollector& cc, const std::string& container);
  bool shouldCopy(const edm::EventSetup& evtSetup, const std::string& container);
  void copyToDB(const edm::EventSetup& evtSetup, const std::string& container);

  std::string m_timetype;
  std::map<std::string, unsigned long long> m_cacheIDs;
  std::map<std::string, std::string> m_records;

  edm::ESGetToken<EcalTPGPedestals, EcalTPGPedestalsRcd> pedestalsToken_;
  edm::ESGetToken<EcalTPGLinearizationConst, EcalTPGLinearizationConstRcd> linearizationConstToken_;
  edm::ESGetToken<EcalTPGSlidingWindow, EcalTPGSlidingWindowRcd> slidingWindowToken_;
  edm::ESGetToken<EcalTPGFineGrainEBIdMap, EcalTPGFineGrainEBIdMapRcd> fineGrainEBIdMapToken_;
  edm::ESGetToken<EcalTPGFineGrainStripEE, EcalTPGFineGrainStripEERcd> fineGrainStripEEToken_;
  edm::ESGetToken<EcalTPGFineGrainTowerEE, EcalTPGFineGrainTowerEERcd> fineGrainTowerEEToken_;
  edm::ESGetToken<EcalTPGLutIdMap, EcalTPGLutIdMapRcd> lutIdMapToken_;
  edm::ESGetToken<EcalTPGWeightIdMap, EcalTPGWeightIdMapRcd> weightIdMapToken_;
  edm::ESGetToken<EcalTPGWeightGroup, EcalTPGWeightGroupRcd> weightGroupToken_;
  edm::ESGetToken<EcalTPGOddWeightIdMap, EcalTPGOddWeightIdMapRcd> oddWeightIdMapToken_;
  edm::ESGetToken<EcalTPGOddWeightGroup, EcalTPGOddWeightGroupRcd> oddWeightGroupToken_;
  edm::ESGetToken<EcalTPGTPMode, EcalTPGTPModeRcd> tpModeToken_;
  edm::ESGetToken<EcalTPGLutGroup, EcalTPGLutGroupRcd> lutGroupToken_;
  edm::ESGetToken<EcalTPGFineGrainEBGroup, EcalTPGFineGrainEBGroupRcd> fineGrainEBGroupToken_;
  edm::ESGetToken<EcalTPGPhysicsConst, EcalTPGPhysicsConstRcd> physicsConstToken_;
  edm::ESGetToken<EcalTPGCrystalStatus, EcalTPGCrystalStatusRcd> crystalStatusToken_;
  edm::ESGetToken<EcalTPGTowerStatus, EcalTPGTowerStatusRcd> towerStatusToken_;
  edm::ESGetToken<EcalTPGSpike, EcalTPGSpikeRcd> spikeToken_;
  edm::ESGetToken<EcalTPGStripStatus, EcalTPGStripStatusRcd> stripStatusToken_;
};

#endif
