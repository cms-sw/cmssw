#ifndef TRIGGERRATESMONITORCLIENT_H
#define TRIGGERRATESMONITORCLIENT_H

//Framework
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/Registry.h"

//DQM
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
 
class TriggerRatesMonitorClient: public DQMEDHarvester{

 public:

  TriggerRatesMonitorClient(const edm::ParameterSet& ps);
  ~TriggerRatesMonitorClient() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      
 protected:

  void beginJob() override;
  void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, edm::LuminosityBlock const &, edm::EventSetup const&) override;  //performed in the endLumi
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;  //performed in the endJob
  
 private:

//private variables
  std::string m_dqm_path;

  // Histograms
  std::vector<TH2F *> m_hltXpd_counts;

};


#endif // TRIGGERRATESMONITORCLIENT_H
