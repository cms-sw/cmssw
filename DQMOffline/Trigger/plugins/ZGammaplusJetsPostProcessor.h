#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "TPRegexp.h"

class ZGammaplusJetsPostProcessor : public DQMEDHarvester {
public:
  ZGammaplusJetsPostProcessor(const edm::ParameterSet &pset);
  ~ZGammaplusJetsPostProcessor() override {}

  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;  // performed in the endJob
  TH1F *histos_(DQMStore::IBooker &ibooker,
                DQMStore::IGetter &igetter,
                const std::string &DBName,
                const std::string &outName,
                const std::string &label,
                const std::string &title);

private:
  std::string subDir_, isMuonTrgigger_, isPhotonTrgigger_;

  TH2F *getHistogram(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter, const std::string &histoPath);
};
