// system include files
#include <iostream>
#include <cstdio>
#include <sys/time.h>

// user include files
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CondFormats/SiStripObjects/interface/SiStripDetVOff.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//
//
// class decleration
//
class SiStripDetVOffReader : public edm::one::EDAnalyzer<> {
public:
  explicit SiStripDetVOffReader(const edm::ParameterSet&);
  ~SiStripDetVOffReader() override = default;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  const bool printdebug_;
  const edm::ESGetToken<SiStripDetVOff, SiStripDetVOffRcd> detVOffToken_;
  std::vector<uint32_t> detids;
};

SiStripDetVOffReader::SiStripDetVOffReader(const edm::ParameterSet& iConfig)
    : printdebug_(iConfig.getUntrackedParameter<bool>("printDebug", true)), detVOffToken_(esConsumes()) {}

void SiStripDetVOffReader::analyze(const edm::Event& e, const edm::EventSetup& iSetup) {
  const auto& detVOff = iSetup.getData(detVOffToken_);
  edm::LogInfo("SiStripDetVOffReader") << "[SiStripDetVOffReader::analyze] End Reading SiStripDetVOff" << std::endl;

  // put here a vector of DetIds to compare
  // Here we just take the vector with all modules that have HV OFF

  // replace this code, with Your own detids
  std::vector<uint32_t> detid;
  detVOff.getDetIds(detid);
  //

  if (printdebug_) {
    for (uint32_t id = 0; id <= detid.size(); id++) {
      bool hvflag = detVOff.IsModuleHVOff(detid[id]);
      bool lvflag = detVOff.IsModuleLVOff(detid[id]);
      bool vflag = detVOff.IsModuleVOff(detid[id]);
      if (hvflag == true) {
        edm::LogInfo("SiStripDetVOffReader") << "detid: " << detid[id] << " HV\t OFF\n";
      } else {
        edm::LogInfo("SiStripDetVOffReader") << "detid: " << detid[id] << " HV\t ON\n";
      }
      if (lvflag == true) {
        edm::LogInfo("SiStripDetVOffReader") << "detid: " << detid[id] << " LV\t OFF\n";
      } else {
        edm::LogInfo("SiStripDetVOffReader") << "detid: " << detid[id] << " LV\t ON\n";
      }
      if (vflag == true) {
        edm::LogInfo("SiStripDetVOffReader") << "detid: " << detid[id] << " V\t OFF\n";
      } else {
        edm::LogInfo("SiStripDetVOffReader") << "detid: " << detid[id] << " V\t ON\n";
      }
    }
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SiStripDetVOffReader);
