#include "HcalTimeSlewEP.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Framework/interface/SourceFactory.h"

#include <string>
#include <vector>
#include <map>

HcalTimeSlewEP::HcalTimeSlewEP(const edm::ParameterSet& pset) : pset_(pset)
{
  setWhatProduced(this);
  findingRecord<HcalTimeSlewRecord>();
}

HcalTimeSlewEP::~HcalTimeSlewEP()
{}

void HcalTimeSlewEP::setIntervalFor(const edm::eventsetup::EventSetupRecordKey& iKey, const edm::IOVSyncValue& iTime, edm::ValidityInterval& oInterval) {
  oInterval = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime()); //infinite
}

void HcalTimeSlewEP::fillDescriptions( edm::ConfigurationDescriptions & descriptions ) {
  //Ask Kevin What goes here??????
  edm::ParameterSetDescription desc;
  desc.add<int>("ieta_shift");
  desc.add<double>("drdA");
  desc.add<double>("drdB");
  edm::ParameterSetDescription desc_dosemaps;
  desc_dosemaps.add<int>("energy");
  desc_dosemaps.add<edm::FileInPath>("file");
  std::vector<edm::ParameterSet> default_dosemap(1);
  desc.addVPSet("dosemaps",desc_dosemaps,default_dosemap);
  edm::ParameterSetDescription desc_years;
  desc_years.add<std::string>("year");
  desc_years.add<double>("intlumi");
  desc_years.add<double>("lumirate");
  desc_years.add<int>("energy");
  std::vector<edm::ParameterSet> default_year(1);
  desc.addVPSet("years",desc_years,default_year);
  
  descriptions.addDefault(desc);
}

// ------------ method called to produce the data  ------------
HcalTimeSlewEP::ReturnType
HcalTimeSlewEP::produce(const HcalTimeSlewRecord& iRecord){

  //Two pset sets for M2 and M3
  std::vector<edm::ParameterSet> p_TimeSlewM2 = pset_.getParameter<std::vector<edm::ParameterSet>>("timeSlewParametersM2");
  std::vector<edm::ParameterSet> p_TimeSlewM3 = pset_.getParameter<std::vector<edm::ParameterSet>>("timeSlewParametersM3");

  ReturnType myResult( new HcalTimeSlew());

  //loop over the VPSets  
  for(const auto& p_timeslew : p_TimeSlewM2){
    double t0       = p_timeslew.getParameter<double>("tzero");
    double m        = p_timeslew.getParameter<double>("slope");
    double tmaximum = p_timeslew.getParameter<double>("tmax");
    myResult->addM2ParameterSet(t0, m, tmaximum);				      
    //dosemaps.emplace(file_energy,HcalTimeSlew::readDoseMap(fp.fullPath()));
  }

  for(const auto& p_timeslew : p_TimeSlewM3){
    double cap_         = p_timeslew.getParameter<double>("cap");
    double tspar0_      = p_timeslew.getParameter<double>("tspar0");
    double tspar1_      = p_timeslew.getParameter<double>("tspar1");
    double tspar2_      = p_timeslew.getParameter<double>("tspar1");
    double tspar0_siPM_ = p_timeslew.getParameter<double>("tspar1");
    double tspar1_siPM_ = p_timeslew.getParameter<double>("tspar1");
    double tspar2_siPM_ = p_timeslew.getParameter<double>("tspar1");
    myResult->addM3ParameterSet(cap_,tspar0_,tspar1_,tspar2_,tspar0_siPM_,tspar1_siPM_,tspar2_siPM_);
  }
  
  return myResult;
}

DEFINE_FWK_EVENTSETUP_SOURCE(HcalTimeSlewEP);
