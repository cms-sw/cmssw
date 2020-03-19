// Migrated to use DQMEDHarvester by: Jyothsna Rani Komaragiri, Oct 2014

#ifndef HLTriggerOffline_JetMET_JetMETDQMPosProcessor_H
#define HLTriggerOffline_JetMET_JetMETDQMPosProcessor_H

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "TEfficiency.h"
#include "TPRegexp.h"

class JetMETDQMPostProcessor : public DQMEDHarvester {
public:
  JetMETDQMPostProcessor(const edm::ParameterSet &pset);
  ~JetMETDQMPostProcessor() override{};

  void dqmEndJob(DQMStore::IBooker &,
                 DQMStore::IGetter &) override;  // performed in the endJob

  TProfile *dividehistos(DQMStore::IBooker &ibooker,
                         DQMStore::IGetter &igetter,
                         const std::string &numName,
                         const std::string &denomName,
                         const std::string &outName,
                         const std::string &label,
                         const std::string &titel);

private:
  std::string subDir_, patternJetTrg_, patternMetTrg_;

  void Efficiency(int passing, int total, double level, double &mode, double &lowerBound, double &upperBound);

  TH1F *getHistogram(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter, const std::string &histoPath);
};

#endif
