#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CalibTracker/SiStripQuality/interface/SiStripQualityWithFromFedErrorsHelper.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"

class SiStripBadStripFromQualityDBWriter : public DQMEDHarvester {
public:
  explicit SiStripBadStripFromQualityDBWriter(const edm::ParameterSet&);
  ~SiStripBadStripFromQualityDBWriter() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void endRun(edm::Run const&, edm::EventSetup const&) override;
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;

private:
  std::string rcdName_, openIOVAt_;
  uint32_t openIOVAtTime_;
  SiStripQualityWithFromFedErrorsHelper withFedErrHelper_;
};

SiStripBadStripFromQualityDBWriter::SiStripBadStripFromQualityDBWriter(const edm::ParameterSet& iConfig)
    : rcdName_{iConfig.getParameter<std::string>("record")},
      openIOVAt_{iConfig.getUntrackedParameter<std::string>("OpenIovAt", "beginOfTime")},
      openIOVAtTime_{iConfig.getUntrackedParameter<uint32_t>("OpenIovAtTime", 1)},
      withFedErrHelper_{iConfig, consumesCollector(), true} {}

void SiStripBadStripFromQualityDBWriter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("record", "");
  desc.addUntracked<std::string>("OpenIovAt", "beginOfTime");
  desc.addUntracked<uint32_t>("OpenIovAtTime", 1);
  SiStripQualityWithFromFedErrorsHelper::fillDescription(desc);
  descriptions.add("siStripBadStripFromQualityDBWriter", desc);
}

void SiStripBadStripFromQualityDBWriter::endRun(edm::Run const& /*run*/, edm::EventSetup const& iSetup) {
  withFedErrHelper_.endRun(iSetup);
}

void SiStripBadStripFromQualityDBWriter::dqmEndJob(DQMStore::IBooker& /*booker*/, DQMStore::IGetter& getter) {
  auto payload = std::make_unique<SiStripBadStrip>(withFedErrHelper_.getMergedQuality(getter));
  cond::Time_t time;
  edm::Service<cond::service::PoolDBOutputService> dbservice;
  if (dbservice.isAvailable()) {
    if (openIOVAt_ == "beginOfTime")
      time = dbservice->beginOfTime();
    else if (openIOVAt_ == "currentTime")
      time = dbservice->currentTime();
    else
      time = openIOVAtTime_;

    dbservice->writeOne(payload.release(), time, rcdName_);
  } else {
    edm::LogError("SiStripBadStripFromQualityDBWriter") << "Service is unavailable" << std::endl;
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
DEFINE_FWK_MODULE(SiStripBadStripFromQualityDBWriter);
