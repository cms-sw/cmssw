#ifndef __L1Trigger_L1THGCal_HGCalTriggerGeoTesterBase_h__
#define __L1Trigger_L1THGCal_HGCalTriggerGeoTesterBase_h__

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TTree.h"
#include <unordered_map>
#include <string>

namespace HepPDT {
  class ParticleDataTable;
}
class MagneticField;
class HGCalTriggerGeometryBase;

struct HGCalTriggerGeoTesterEventSetup {
  edm::ESHandle<HGCalTriggerGeometryBase> geometry;
};

class HGCalTriggerGeoTesterErrors {
public:
  enum ErrorCode {
    CellValidity = 0,
    MissingCellInTC,
    InvalidCellInTC,
    MissingTCInModule,
    InvalidTCInModule,
    ConnectedModuleWithoutLpgbt,
    ModuleSplitInStage1,
    MissingModuleInStage1FPGA,
    InvalidModuleInStage1FPGA,
    MissingStage1InStage2FPGA,
    InvalidStage1InStage2FPGA
  };
  static const std::unordered_map<ErrorCode, std::string> messages;

  void fill(ErrorCode error, unsigned detid) {
    auto itr_err = error_detids_.insert({error, {}}).first;
    itr_err->second.insert(detid);
    auto itr_id = detid_errors_.insert({detid, {}}).first;
    itr_id->second.insert(error);
  }
  const std::unordered_map<ErrorCode, std::set<unsigned>>& errors() const { return error_detids_; }
  const std::unordered_map<unsigned, std::set<ErrorCode>>& detids() const { return detid_errors_; }

private:
  std::unordered_map<ErrorCode, std::set<unsigned>> error_detids_;
  std::unordered_map<unsigned, std::set<ErrorCode>> detid_errors_;
};

class HGCalTriggerGeoTesterBase {
public:
  HGCalTriggerGeoTesterBase(const edm::ParameterSet& conf) : name_(conf.getParameter<std::string>("TesterName")) {};
  virtual ~HGCalTriggerGeoTesterBase() {};
  const std::string& name() const { return name_; }
  virtual void initialize(TTree*, const edm::ParameterSet&) = 0;
  virtual void check(const HGCalTriggerGeoTesterEventSetup&) = 0;
  virtual void fill(const HGCalTriggerGeoTesterEventSetup&) = 0;
  const HGCalTriggerGeoTesterErrors& errors() { return errors_; }

protected:
  virtual void clear() = 0;
  const std::string name_;
  HGCalTriggerGeoTesterErrors errors_;
  TTree* tree_;
};

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory<HGCalTriggerGeoTesterBase*(const edm::ParameterSet&)> HGCalTriggerGeoTesterFactory;

#endif
