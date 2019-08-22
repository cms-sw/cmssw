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

HcalTimeSlewEP::HcalTimeSlewEP(const edm::ParameterSet& pset) : pset_(pset) {
  setWhatProduced(this);
  findingRecord<HcalTimeSlewRecord>();
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
  //Two pset sets for M2/Simulation and M3
  std::vector<edm::ParameterSet> p_TimeSlewM2 =
      pset_.getParameter<std::vector<edm::ParameterSet>>("timeSlewParametersM2");
  std::vector<edm::ParameterSet> p_TimeSlewM3 =
      pset_.getParameter<std::vector<edm::ParameterSet>>("timeSlewParametersM3");

  ReturnType hcalTimeSlew = std::make_unique<HcalTimeSlew>();

  //loop over the VPSets
  for (const auto& p_timeslew : p_TimeSlewM2) {
    float t0 = p_timeslew.getParameter<double>("tzero");
    float m = p_timeslew.getParameter<double>("slope");
    float tmaximum = p_timeslew.getParameter<double>("tmax");
    hcalTimeSlew->addM2ParameterSet(t0, m, tmaximum);
  }

  for (const auto& p_timeslew : p_TimeSlewM3) {
    double cap_ = p_timeslew.getParameter<double>("cap");
    double tspar0_ = p_timeslew.getParameter<double>("tspar0");
    double tspar1_ = p_timeslew.getParameter<double>("tspar1");
    double tspar2_ = p_timeslew.getParameter<double>("tspar2");
    double tspar0_siPM_ = p_timeslew.getParameter<double>("tspar0_siPM");
    double tspar1_siPM_ = p_timeslew.getParameter<double>("tspar1_siPM");
    double tspar2_siPM_ = p_timeslew.getParameter<double>("tspar2_siPM");
    hcalTimeSlew->addM3ParameterSet(cap_, tspar0_, tspar1_, tspar2_, tspar0_siPM_, tspar1_siPM_, tspar2_siPM_);
  }

  return hcalTimeSlew;
}

DEFINE_FWK_EVENTSETUP_SOURCE(HcalTimeSlewEP);
