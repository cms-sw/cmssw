#ifndef SusyPostProcessor_H
#define SusyPostProcessor_H

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include <vector>
#include <string>

#include "TFile.h"
#include "TH1.h"
#include "TMath.h"

class SusyPostProcessor : public DQMEDHarvester
{
 public:
  explicit SusyPostProcessor( const edm::ParameterSet& pSet ) ;
  ~SusyPostProcessor();
                                   

 private:

  edm::ParameterSet iConfig;
  virtual void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) ;
  void QuantilePlots(MonitorElement* &, double, DQMStore::IBooker &);

  static const char* messageLoggerCatregory;

  std::string SUSYFolder;
  double _quantile;

  std::vector<MonitorElement*> histoVector;
  std::vector<std::string> Dirs;

  MonitorElement* MEx;
  MonitorElement* MEy;
};

#endif
