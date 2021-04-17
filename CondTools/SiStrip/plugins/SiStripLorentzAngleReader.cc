#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"

#include "CondTools/SiStrip/plugins/SiStripLorentzAngleReader.h"

#include <iostream>
#include <cstdio>
#include <sys/time.h>

using namespace cms;

SiStripLorentzAngleReader::SiStripLorentzAngleReader(const edm::ParameterSet& iConfig)
    : printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug", 5)),
      label_(iConfig.getUntrackedParameter<std::string>("label", "")),
      laToken_(esConsumes(edm::ESInputTag{"", label_})) {}
SiStripLorentzAngleReader::~SiStripLorentzAngleReader() {}

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
