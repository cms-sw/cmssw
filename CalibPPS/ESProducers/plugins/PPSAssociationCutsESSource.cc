/****************************************************************************
 * Authors:
 *   Jan Kašpar
 *   Grzegorz Sroka
 ****************************************************************************/

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESProducts.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/PPSObjects/interface/PPSAssociationCuts.h"
#include "CondFormats/DataRecord/interface/PPSAssociationCutsRcd.h"
#include <vector>
#include <optional>

using namespace std;

class PPSAssociationCutsESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  PPSAssociationCutsESSource(const edm::ParameterSet &);

  ~PPSAssociationCutsESSource() override = default;

  std::shared_ptr<PPSAssociationCuts> produce(const PPSAssociationCutsRcd &);

  static void fillDescriptions(edm::ConfigurationDescriptions &);

protected:
  bool isConcurrentFinder() const override { return true; }

  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &,
                      const edm::IOVSyncValue &,
                      edm::ValidityInterval &) override;

private:
  std::optional<unsigned int> associationCutIndexFor(const edm::IOVSyncValue &) const;
  static edm::ParameterSetDescription getIOVDefaultParameters();
  bool currentAssociationCutValid_;
  unsigned int currentAssociationCutIdx_;
  std::vector<std::shared_ptr<PPSAssociationCuts>> ppsAssociationCuts_;
  std::vector<edm::EventRange> validityRanges_;
};

//----------------------------------------------------------------------------------------------------

PPSAssociationCutsESSource::PPSAssociationCutsESSource(const edm::ParameterSet &iConfig) {
  for (const auto &interval : iConfig.getParameter<std::vector<edm::ParameterSet>>("configuration")) {
    ppsAssociationCuts_.push_back(make_shared<PPSAssociationCuts>(interval));
    validityRanges_.push_back(interval.getParameter<edm::EventRange>("validityRange"));
  }

  setWhatProduced(this);
  findingRecord<PPSAssociationCutsRcd>();
}

//----------------------------------------------------------------------------------------------------

std::optional<unsigned int> PPSAssociationCutsESSource::associationCutIndexFor(const edm::IOVSyncValue &iosv) const {
  for (unsigned int idx = 0; idx < ppsAssociationCuts_.size(); ++idx) {
    // is within an entry ?
    if (edm::contains(validityRanges_[idx], iosv.eventID())) {
      return idx;
    }
  }
  return std::nullopt;
}

void PPSAssociationCutsESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &key,
                                                const edm::IOVSyncValue &iosv,
                                                edm::ValidityInterval &oValidity) {
  auto cutIdx = associationCutIndexFor(iosv);
  if (cutIdx) {
    oValidity = edm::ValidityInterval(edm::IOVSyncValue(validityRanges_[*cutIdx].startEventID()),
                                      edm::IOVSyncValue(validityRanges_[*cutIdx].endEventID()));
    return;
  }

  edm::LogInfo("PPSAssociationCutsESSource")
      << ">> PPSAssociationCutsESSource::setIntervalFor(" << key.name() << ")\n"
      << "    run=" << iosv.eventID().run() << ", event=" << iosv.eventID().event();

  const edm::EventID start(iosv.eventID().run(), iosv.eventID().luminosityBlock(), iosv.eventID().event());
  const edm::EventID end(iosv.eventID().run(), iosv.eventID().luminosityBlock(), iosv.eventID().event());
  oValidity = edm::ValidityInterval(edm::IOVSyncValue(start), edm::IOVSyncValue(end));
}

//----------------------------------------------------------------------------------------------------

std::shared_ptr<PPSAssociationCuts> PPSAssociationCutsESSource::produce(const PPSAssociationCutsRcd &iRcd) {
  auto output = std::make_shared<PPSAssociationCuts>();

  auto cutIdx = associationCutIndexFor(iRcd.validityInterval().first());

  if (cutIdx) {
    const auto &associationCut = ppsAssociationCuts_[*cutIdx];
    output = associationCut;
  }

  return output;
}

//----------------------------------------------------------------------------------------------------

void PPSAssociationCutsESSource::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("ppsAssociationCutsLabel", "");

  edm::ParameterSetDescription validator = PPSAssociationCutsESSource::getIOVDefaultParameters();

  std::vector<edm::ParameterSet> vDefaults;
  desc.addVPSet("configuration", validator, vDefaults);

  descriptions.add("ppsAssociationCutsESSource", desc);
}

edm::ParameterSetDescription PPSAssociationCutsESSource::getIOVDefaultParameters() {
  edm::ParameterSetDescription desc;
  desc.add<edm::EventRange>("validityRange", edm::EventRange())->setComment("interval of validity");

  for (auto &sector : {"45", "56"}) {
    desc.add<edm::ParameterSetDescription>("association_cuts_" + std::string(sector),
                                           PPSAssociationCuts::getDefaultParameters())
        ->setComment("track-association cuts for sector " + std::string(sector));
  }

  return desc;
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_EVENTSETUP_SOURCE(PPSAssociationCutsESSource);
