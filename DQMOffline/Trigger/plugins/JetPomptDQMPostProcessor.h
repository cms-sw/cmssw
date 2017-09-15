// Migrated to use DQMEDHarvester by: Jyothsna Rani Komaragiri, Oct 2014

#ifndef DQMOffline_Trigger_JetDQMPosProcessor_H
#define DQMOffline_Trigger_JetDQMPosProcessor_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include "TPRegexp.h"
#include "TEfficiency.h"

class JetPomptDQMPostProcessor : public DQMEDHarvester{
 public:
  JetPomptDQMPostProcessor(const edm::ParameterSet& pset);
  ~JetPomptDQMPostProcessor() {};

  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override; //performed in the endJob
  
    void dividehistos(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter, const std::string& numName, const std::string& denomName, const std::string& outName, const std::string& label, const std::string& titel, const std::string histDim);

 private:
  std::string subDir_;

  TH1F *getHistogram(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter, const std::string &histoPath);
  TH2F *getHistogram2D(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter, const std::string &histoPath);

};

#endif
