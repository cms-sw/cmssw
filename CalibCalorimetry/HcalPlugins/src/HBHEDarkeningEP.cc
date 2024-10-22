#include "HBHEDarkeningEP.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Framework/interface/SourceFactory.h"

#include <string>
#include <vector>
#include <map>

HBHEDarkeningEP::HBHEDarkeningEP(const edm::ParameterSet& pset)
    : drdA_(pset.getParameter<double>("drdA")),
      drdB_(pset.getParameter<double>("drdB")),
      ieta_shift_(pset.getParameter<int>("ieta_shift")) {
  setWhatProduced(this);
  findingRecord<HBHEDarkeningRecord>();

  const std::vector<edm::ParameterSet>& p_dosemaps = pset.getParameter<std::vector<edm::ParameterSet>>("dosemaps");
  dosemaps_.reserve(p_dosemaps.size());
  for (const auto& p_dosemap : p_dosemaps) {
    dosemaps_.push_back({p_dosemap.getParameter<edm::FileInPath>("file"), p_dosemap.getParameter<int>("energy")});
  }

  //initialize years
  const std::vector<edm::ParameterSet>& p_years = pset.getParameter<std::vector<edm::ParameterSet>>("years");
  years_.reserve(p_years.size());
  for (const auto& p_year : p_years) {
    years_.emplace_back(p_year.getParameter<std::string>("year"),
                        p_year.getParameter<double>("intlumi"),
                        p_year.getParameter<double>("lumirate"),
                        p_year.getParameter<int>("energy"));
  }
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
  std::map<int, std::vector<std::vector<float>>> dosemaps;
  for (const auto& p_dosemap : dosemaps_) {
    dosemaps.emplace(p_dosemap.file_energy, HBHEDarkening::readDoseMap(p_dosemap.fp.fullPath()));
  }

  return std::make_unique<HBHEDarkening>(ieta_shift_, drdA_, drdB_, std::move(dosemaps), years_);
}

DEFINE_FWK_EVENTSETUP_SOURCE(HBHEDarkeningEP);
