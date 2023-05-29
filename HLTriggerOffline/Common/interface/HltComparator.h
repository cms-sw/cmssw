#ifndef HLTCOMPARATOR_H
#define HLTCOMPARATOR_H
// Original Author: James Jackson

#include "FWCore/Framework/interface/one/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/TriggerResults.h"

class TH1F;

class HltComparator : public edm::one::EDFilter<edm::one::SharedResources> {
public:
  explicit HltComparator(const edm::ParameterSet &);
  ~HltComparator() override;

private:
  edm::EDGetTokenT<edm::TriggerResults> hltOnlineResults_;
  edm::EDGetTokenT<edm::TriggerResults> hltOfflineResults_;

  std::vector<std::string> onlineActualNames_;
  std::vector<std::string> offlineActualNames_;
  std::vector<unsigned int> onlineToOfflineBitMappings_;

  std::vector<TH1F *> comparisonHists_;
  std::map<unsigned int, std::map<std::string, unsigned int>> triggerComparisonErrors_;

  bool init_;
  bool verbose_;
  bool verbose() const { return verbose_; }

  std::vector<std::string> skipPathList_;
  std::vector<std::string> usePathList_;

  unsigned int numTriggers_;

  void beginJob() override;
  bool filter(edm::Event &, const edm::EventSetup &) override;
  void endJob() override;
  void initialise(const edm::TriggerResults &, const edm::TriggerResults &, edm::Event &e);
  std::string formatResult(const unsigned int);
};

#endif  // HLTCOMPARATOR_HH
