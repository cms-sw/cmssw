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
  // HcalTimeSlewEP
  edm::ParameterSetDescription desc;
  desc.add<std::string>("appendToDataLabel", "HBHE");
  {
    edm::ParameterSetDescription vpsd1;
    vpsd1.add<double>("tzero", 23.960177);
    vpsd1.add<double>("slope", -3.178648);
    vpsd1.add<double>("tmax", 16.0);
    std::vector<edm::ParameterSet> temp1;
    temp1.reserve(3);
    {
      edm::ParameterSet temp2;
      temp2.addParameter<double>("tzero", 23.960177);
      temp2.addParameter<double>("slope", -3.178648);
      temp2.addParameter<double>("tmax", 16.0);
      temp1.push_back(temp2);
    }
    {
      edm::ParameterSet temp2;
      temp2.addParameter<double>("tzero", 11.977461);
      temp2.addParameter<double>("slope", -1.5610227);
      temp2.addParameter<double>("tmax", 10.0);
      temp1.push_back(temp2);
    }
    {
      edm::ParameterSet temp2;
      temp2.addParameter<double>("tzero", 9.109694);
      temp2.addParameter<double>("slope", -1.075824);
      temp2.addParameter<double>("tmax", 6.25);
      temp1.push_back(temp2);
    }
    desc.addVPSet("timeSlewParametersM2", vpsd1, temp1);
  }
  {
    edm::ParameterSetDescription vpsd1;
    vpsd1.add<double>("cap", 6.0);
    vpsd1.add<double>("tspar0", 12.2999);
    vpsd1.add<double>("tspar1", -2.19142);
    vpsd1.add<double>("tspar2", 0.0);
    vpsd1.add<double>("tspar0_siPM", 0.0);
    vpsd1.add<double>("tspar1_siPM", 0.0);
    vpsd1.add<double>("tspar2_siPM", 0.0);
    std::vector<edm::ParameterSet> temp1;
    temp1.reserve(4);
    {
      edm::ParameterSet temp2;
      temp2.addParameter<double>("cap", 6.0);
      temp2.addParameter<double>("tspar0", 12.2999);
      temp2.addParameter<double>("tspar1", -2.19142);
      temp2.addParameter<double>("tspar2", 0.0);
      temp2.addParameter<double>("tspar0_siPM", 0.0);
      temp2.addParameter<double>("tspar1_siPM", 0.0);
      temp2.addParameter<double>("tspar2_siPM", 0.0);
      temp1.push_back(temp2);
    }
    {
      edm::ParameterSet temp2;
      temp2.addParameter<double>("cap", 6.0);
      temp2.addParameter<double>("tspar0", 15.5);
      temp2.addParameter<double>("tspar1", -3.2);
      temp2.addParameter<double>("tspar2", 32.0);
      temp2.addParameter<double>("tspar0_siPM", 0.0);
      temp2.addParameter<double>("tspar1_siPM", 0.0);
      temp2.addParameter<double>("tspar2_siPM", 0.0);
      temp1.push_back(temp2);
    }
    {
      edm::ParameterSet temp2;
      temp2.addParameter<double>("cap", 6.0);
      temp2.addParameter<double>("tspar0", 12.2999);
      temp2.addParameter<double>("tspar1", -2.19142);
      temp2.addParameter<double>("tspar2", 0.0);
      temp2.addParameter<double>("tspar0_siPM", 0.0);
      temp2.addParameter<double>("tspar1_siPM", 0.0);
      temp2.addParameter<double>("tspar2_siPM", 0.0);
      temp1.push_back(temp2);
    }
    {
      edm::ParameterSet temp2;
      temp2.addParameter<double>("cap", 6.0);
      temp2.addParameter<double>("tspar0", 12.2999);
      temp2.addParameter<double>("tspar1", -2.19142);
      temp2.addParameter<double>("tspar2", 0.0);
      temp2.addParameter<double>("tspar0_siPM", 0.0);
      temp2.addParameter<double>("tspar1_siPM", 0.0);
      temp2.addParameter<double>("tspar2_siPM", 0.0);
      temp1.push_back(temp2);
    }
    desc.addVPSet("timeSlewParametersM3", vpsd1, temp1);
  }
  descriptions.addWithDefaultLabel(desc);
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
