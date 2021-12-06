// system include files
#include <iostream>
#include <cstdio>
#include <sys/time.h>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/DataRecord/interface/SiStripLorentzAngleRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"

//
//
// class decleration
//
class SiStripLorentzAngleReader : public edm::one::EDAnalyzer<> {
public:
  explicit SiStripLorentzAngleReader(const edm::ParameterSet&);
  ~SiStripLorentzAngleReader() override = default;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  uint32_t printdebug_;
  std::string label_;
  const edm::ESGetToken<SiStripLorentzAngle, SiStripLorentzAngleRcd> laToken_;
};

using namespace cms;

SiStripLorentzAngleReader::SiStripLorentzAngleReader(const edm::ParameterSet& iConfig)
    : printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug", 5)),
      label_(iConfig.getUntrackedParameter<std::string>("label", "")),
      laToken_(esConsumes(edm::ESInputTag{"", label_})) {}

void SiStripLorentzAngleReader::analyze(const edm::Event& e, const edm::EventSetup& iSetup) {
  const auto& lorentzAngles = iSetup.getData(laToken_);
  edm::LogInfo("SiStripLorentzAngleReader")
      << "[SiStripLorentzAngleReader::analyze] End Reading SiStripLorentzAngle with label " << label_ << std::endl;

  std::map<unsigned int, float> detid_la = lorentzAngles.getLorentzAngles();
  std::map<unsigned int, float>::const_iterator it;
  size_t count = 0;
  for (it = detid_la.begin(); it != detid_la.end() && count < printdebug_; it++) {
    edm::LogInfo("SiStripLorentzAngleReader") << "detid " << it->first << " \t"
                                              << " Lorentz angle  " << it->second;
    count++;
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SiStripLorentzAngleReader);
