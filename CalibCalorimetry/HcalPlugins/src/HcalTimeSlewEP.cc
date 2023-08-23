#include "HcalTimeSlewEP.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Framework/interface/SourceFactory.h"

#include <string>
#include <vector>
#include <map>
#include <iostream>

HcalTimeSlewEP::HcalTimeSlewEP(const edm::ParameterSet& pset) {
  setWhatProduced(this);
  findingRecord<HcalTimeSlewRecord>();

  //Two pset sets for M2/Simulation and M3
  std::vector<edm::ParameterSet> p_TimeSlewM2 =
      pset.getParameter<std::vector<edm::ParameterSet>>("timeSlewParametersM2");
  std::vector<edm::ParameterSet> p_TimeSlewM3 =
      pset.getParameter<std::vector<edm::ParameterSet>>("timeSlewParametersM3");

  //loop over the VPSets
  for (const auto& p_timeslew : p_TimeSlewM2) {
    m2parameters_.push_back({static_cast<float>(p_timeslew.getParameter<double>("tzero")),
                             static_cast<float>(p_timeslew.getParameter<double>("slope")),
                             static_cast<float>(p_timeslew.getParameter<double>("tmax"))});
  }

  for (const auto& p_timeslew : p_TimeSlewM3) {
    m3parameters_.push_back({p_timeslew.getParameter<double>("cap"),
                             p_timeslew.getParameter<double>("tspar0"),
                             p_timeslew.getParameter<double>("tspar1"),
                             p_timeslew.getParameter<double>("tspar2"),
                             p_timeslew.getParameter<double>("tspar0_siPM"),
                             p_timeslew.getParameter<double>("tspar1_siPM"),
                             p_timeslew.getParameter<double>("tspar2_siPM")});
  }
}

HcalTimeSlewEP::~HcalTimeSlewEP() {}

void HcalTimeSlewEP::setIntervalFor(const edm::eventsetup::EventSetupRecordKey& iKey,
                                    const edm::IOVSyncValue& iTime,
                                    edm::ValidityInterval& oInterval) {
  oInterval = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
}

void HcalTimeSlewEP::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  edm::ParameterSetDescription desc_M2;
  desc_M2.add<double>("tzero");
  desc_M2.add<double>("slope");
  desc_M2.add<double>("tmax");
  std::vector<edm::ParameterSet> default_M2(1);
  desc.addVPSet("timeSlewParametersM2", desc_M2, default_M2);

  edm::ParameterSetDescription desc_M3;
  desc_M3.add<double>("cap");
  desc_M3.add<double>("tspar0");
  desc_M3.add<double>("tspar1");
  desc_M3.add<double>("tspar2");
  desc_M3.add<double>("tspar0_siPM");
  desc_M3.add<double>("tspar1_siPM");
  desc_M3.add<double>("tspar2_siPM");
  std::vector<edm::ParameterSet> default_M3(1);
  desc.addVPSet("timeSlewParametersM3", desc_M3, default_M3);

  descriptions.addDefault(desc);
}

// ------------ method called to produce the data  ------------
HcalTimeSlewEP::ReturnType HcalTimeSlewEP::produce(const HcalTimeSlewRecord& iRecord) {
  ReturnType hcalTimeSlew = std::make_unique<HcalTimeSlew>();

  //loop over the VPSets
  for (const auto& p : m2parameters_) {
    hcalTimeSlew->addM2ParameterSet(p.t0, p.m, p.tmaximum);
  }

  for (const auto& p : m3parameters_) {
    hcalTimeSlew->addM3ParameterSet(p.cap, p.tspar0, p.tspar1, p.tspar2, p.tspar0_siPM, p.tspar1_siPM, p.tspar2_siPM);
  }

  return hcalTimeSlew;
}

DEFINE_FWK_EVENTSETUP_SOURCE(HcalTimeSlewEP);
