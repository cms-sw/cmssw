#ifndef DQMCORRELATIONCLIENT_H
#define DQMCORRELATIONCLIENT_H

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
#include "DQMServices/Core/interface/MonitorElement.h"
 
struct MEPSet {
  std::string name;
  std::string folder;
  bool        profileX;
};

struct OutputMEPSet {
  std::string name;
  std::string folder;
  bool doXaxis;
  int nbinsX;
  double xminX;
  double xmaxX;
  bool doYaxis;
  int nbinsY;
  double xminY;
  double xmaxY;
};

class DQMCorrelationClient: public DQMEDHarvester{

 public:

  DQMCorrelationClient(const edm::ParameterSet& ps);
  virtual ~DQMCorrelationClient() = default;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  static void fillMePSetDescription(edm::ParameterSetDescription & pset);
  static void fillOutputMePSetDescription(edm::ParameterSetDescription & pset);
      
 protected:

  void beginJob() override;
  void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, edm::LuminosityBlock const &, edm::EventSetup const&) override;  //performed in the endLumi
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;  //performed in the endJob
  
 private:

  static MEPSet       getHistoPSet      (edm::ParameterSet pset);
  static OutputMEPSet getOutputHistoPSet(edm::ParameterSet pset);

  TH1* getTH1(MonitorElement* me, bool profileX);
  void setAxisTitle(MonitorElement* meX, MonitorElement* meY);

//private variables

  //variables from config file
  bool me1onX_;

  // Histograms
  MonitorElement* correlation_;

  MEPSet meXpset_;
  MEPSet meYpset_;
  OutputMEPSet mepset_;

};


#endif // DQMCORRELATIONCLIENT_H
