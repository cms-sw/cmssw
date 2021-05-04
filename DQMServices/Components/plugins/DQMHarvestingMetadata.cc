#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include "FWCore/Version/interface/GetReleaseVersion.h"

class DQMHarvestingMetadata : public DQMEDHarvester {
public:
  /// Constructor
  DQMHarvestingMetadata(const edm::ParameterSet& ps);

  /// Destructor
  ~DQMHarvestingMetadata() override;

protected:
  /// Analyze
  void dqmEndRun(DQMStore::IBooker& ibooker,
                 DQMStore::IGetter& igetter,
                 edm::Run const& iRun,
                 edm::EventSetup const& /* iSetup */) override;
  void dqmEndLuminosityBlock(DQMStore::IBooker& ibooker,
                             DQMStore::IGetter& igetter,
                             edm::LuminosityBlock const& iLumi,
                             edm::EventSetup const& /* iSetup */) override;

  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override{};

private:
  std::string eventInfoFolder_;
  std::string subsystemname_;

  MonitorElement* runId_;
  MonitorElement* runList_;
  MonitorElement* runStartTimeStamp_;  ///UTC time of the run start
  MonitorElement* lumisecId_;
  MonitorElement* firstLumisecId_;
  MonitorElement* lastLumisecId_;

  MonitorElement* processTimeStamp_;  ///The UTC time of the job initialization
  MonitorElement* hostName_;          ///Hostname of the local machine
  MonitorElement* workingDir_;        ///Current working directory of the job
  MonitorElement* cmsswVer_;          ///CMSSW version run for this job

  double currentTime_;
};

static inline double stampToReal(edm::Timestamp time) {
  return (time.value() >> 32) + 1e-6 * (time.value() & 0xffffffff);
}

static inline double stampToReal(const timeval& time) { return time.tv_sec + 1e-6 * time.tv_usec; }

DQMHarvestingMetadata::DQMHarvestingMetadata(const edm::ParameterSet& ps) {
  struct timeval now;
  gettimeofday(&now, nullptr);
  currentTime_ = stampToReal(now);

  // read config parms
  std::string folder = ps.getUntrackedParameter<std::string>("eventInfoFolder", "EventInfo");
  subsystemname_ = ps.getUntrackedParameter<std::string>("subSystemFolder", "YourSubsystem");

  eventInfoFolder_ = subsystemname_ + "/" + folder;
}

DQMHarvestingMetadata::~DQMHarvestingMetadata() = default;

void DQMHarvestingMetadata::dqmEndRun(DQMStore::IBooker& ibooker,
                                      DQMStore::IGetter& igetter,
                                      edm::Run const& iRun,
                                      edm::EventSetup const& /* iSetup */) {
  ibooker.setCurrentFolder(eventInfoFolder_);

  runList_ = ibooker.bookString("Run", "");
  runId_ = ibooker.bookInt("iRun");
  runStartTimeStamp_ = ibooker.bookFloat("runStartTimeStamp");

  if (runList_->getStringValue().empty()) {
    std::string run = std::to_string(iRun.id().run());
    runList_->Fill(run);
    runId_->Fill(iRun.id().run());
    // in case of multiple runs, record start time of the first.
    runStartTimeStamp_->Fill(stampToReal(iRun.beginTime()));
  } else {
    std::string run = runList_->getStringValue() + "," + std::to_string(iRun.id().run());
    runList_->Fill(run);
    // this is the agreed-upon pseudo-runnumber for multi-run harvesting.
    runId_->Fill(999999);
  }

  processTimeStamp_ = ibooker.bookFloat("processTimeStamp");
  processTimeStamp_->Fill(currentTime_);

  char hostname[65];
  gethostname(hostname, 64);
  hostname[64] = 0;
  hostName_ = ibooker.bookString("hostName", hostname);
  char* pwd = getcwd(nullptr, 0);
  workingDir_ = ibooker.bookString("workingDir", pwd);
  free(pwd);
  cmsswVer_ = ibooker.bookString("CMSSW_Version", edm::getReleaseVersion());
}

void DQMHarvestingMetadata::dqmEndLuminosityBlock(DQMStore::IBooker& ibooker,
                                                  DQMStore::IGetter& igetter,
                                                  edm::LuminosityBlock const& iLumi,
                                                  edm::EventSetup const& /* iSetup */) {
  int lumi = iLumi.luminosityBlock();

  ibooker.setCurrentFolder(eventInfoFolder_);
  firstLumisecId_ = ibooker.bookInt("firstLumiSection");
  lastLumisecId_ = ibooker.bookInt("lastLumiSection");
  lumisecId_ = ibooker.bookInt("iLumiSection");
  lumisecId_->Fill(lumi);

  if (firstLumisecId_->getIntValue() == 0 || firstLumisecId_->getIntValue() > lumi) {
    firstLumisecId_->Fill(lumi);
  }
  if (lastLumisecId_->getIntValue() == 0 || lastLumisecId_->getIntValue() > lumi) {
    lastLumisecId_->Fill(lumi);
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DQMHarvestingMetadata);
