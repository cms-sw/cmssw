#include "DQMOffline/Hcal/interface/HcalNoiseRatesClient.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"

HcalNoiseRatesClient::HcalNoiseRatesClient(const edm::ParameterSet &iConfig) : conf_(iConfig) {
  outputFile_ = iConfig.getUntrackedParameter<std::string>("outputFile", "myfile.root");
  debug_ = false;
  verbose_ = false;
  dirName_ = iConfig.getParameter<std::string>("DQMDirName");
}

HcalNoiseRatesClient::~HcalNoiseRatesClient() {}

void HcalNoiseRatesClient::beginJob() {}

void HcalNoiseRatesClient::beginRun(const edm::Run &run, const edm::EventSetup &c) {}

void HcalNoiseRatesClient::dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {
  igetter.setCurrentFolder(dirName_);

  if (verbose_)
    std::cout << "\nrunClient" << std::endl;

  std::vector<MonitorElement *> hcalMEs;

  // Since out folders are fixed to three, we can just go over these three
  // folders i.e., CaloTowersD/CaloTowersTask, HcalRecHitsD/HcalRecHitTask,
  // HcalNoiseRatesD/NoiseRatesTask.
  std::vector<std::string> fullPathHLTFolders = igetter.getSubdirs();
  for (const auto &fullPathHLTFolder : fullPathHLTFolders) {
    if (verbose_)
      std::cout << "\nfullPath: " << fullPathHLTFolder << std::endl;
    igetter.setCurrentFolder(fullPathHLTFolder);

    std::vector<std::string> fullSubPathHLTFolders = igetter.getSubdirs();
    for (auto &fullSubPathHLTFolder : fullSubPathHLTFolders) {
      if (verbose_)
        std::cout << "fullSub: " << fullSubPathHLTFolder << std::endl;

      if (strcmp(fullSubPathHLTFolder.c_str(), "HcalNoiseRatesD/NoiseRatesTask") == 0) {
        hcalMEs = igetter.getContents(fullSubPathHLTFolder);
        if (verbose_)
          std::cout << "hltMES size : " << hcalMEs.size() << std::endl;
        if (!NoiseRatesEndjob(hcalMEs))
          std::cout << "\nError in NoiseRatesEndjob!" << std::endl << std::endl;
      }
    }
  }
}

// called after entering the HcalNoiseRatesD/NoiseRatesTask directory
// hcalMEs are within that directory
int HcalNoiseRatesClient::NoiseRatesEndjob(const std::vector<MonitorElement *> &hcalMEs) {
  int useAllHistos = 0;
  MonitorElement *hLumiBlockCount = nullptr;
  for (auto hcalME : hcalMEs) {
    if (strcmp(hcalME->getName().c_str(), "hLumiBlockCount") == 0) {
      hLumiBlockCount = hcalME;
      useAllHistos = 1;
    }
  }
  if (useAllHistos != 0 && useAllHistos != 1)
    return 0;

  // FIXME: dummy lumiCountMap.size since hLumiBlockCount is disabled
  // in a general case.
  int lumiCountMapsize = -1;  // dummy
  if (useAllHistos)
    hLumiBlockCount->Fill(0.0, lumiCountMapsize);

  return 1;
}

DEFINE_FWK_MODULE(HcalNoiseRatesClient);
