#include "HBHEDarkeningEP.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Framework/interface/SourceFactory.h"

#include <string>
#include <vector>
#include <map>

HBHEDarkeningEP::HBHEDarkeningEP(const edm::ParameterSet& pset) : pset_(pset) {
  setWhatProduced(this);
  findingRecord<HBHEDarkeningRecord>();
}

HBHEDarkeningEP::~HBHEDarkeningEP() {}

void HBHEDarkeningEP::setIntervalFor(const edm::eventsetup::EventSetupRecordKey& iKey,
                                     const edm::IOVSyncValue& iTime,
                                     edm::ValidityInterval& oInterval) {
  oInterval = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());  //infinite
}

void HBHEDarkeningEP::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<int>("ieta_shift");
  desc.add<double>("drdA");
  desc.add<double>("drdB");
  edm::ParameterSetDescription desc_dosemaps;
  desc_dosemaps.add<int>("energy");
  desc_dosemaps.add<edm::FileInPath>("file");
  std::vector<edm::ParameterSet> default_dosemap(1);
  desc.addVPSet("dosemaps", desc_dosemaps, default_dosemap);
  edm::ParameterSetDescription desc_years;
  desc_years.add<std::string>("year");
  desc_years.add<double>("intlumi");
  desc_years.add<double>("lumirate");
  desc_years.add<int>("energy");
  std::vector<edm::ParameterSet> default_year(1);
  desc.addVPSet("years", desc_years, default_year);

  descriptions.addDefault(desc);
}

// ------------ method called to produce the data  ------------
HBHEDarkeningEP::ReturnType HBHEDarkeningEP::produce(const HBHEDarkeningRecord& iRecord) {
  //initialize dose maps
  std::vector<edm::ParameterSet> p_dosemaps = pset_.getParameter<std::vector<edm::ParameterSet>>("dosemaps");
  std::map<int, std::vector<std::vector<float>>> dosemaps;
  for (const auto& p_dosemap : p_dosemaps) {
    edm::FileInPath fp = p_dosemap.getParameter<edm::FileInPath>("file");
    int file_energy = p_dosemap.getParameter<int>("energy");
    dosemaps.emplace(file_energy, HBHEDarkening::readDoseMap(fp.fullPath()));
  }

  //initialize years
  std::vector<edm::ParameterSet> p_years = pset_.getParameter<std::vector<edm::ParameterSet>>("years");
  std::vector<HBHEDarkening::LumiYear> years;
  years.reserve(p_years.size());
  for (const auto& p_year : p_years) {
    years.emplace_back(p_year.getParameter<std::string>("year"),
                       p_year.getParameter<double>("intlumi"),
                       p_year.getParameter<double>("lumirate"),
                       p_year.getParameter<int>("energy"));
  }

  return std::make_unique<HBHEDarkening>(pset_.getParameter<int>("ieta_shift"),
                                         pset_.getParameter<double>("drdA"),
                                         pset_.getParameter<double>("drdB"),
                                         std::move(dosemaps),
                                         std::move(years));
}

DEFINE_FWK_EVENTSETUP_SOURCE(HBHEDarkeningEP);
