#ifndef DQMSCALETOCLIENT_H
#define DQMSCALETOCLIENT_H

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
};

struct OutputMEPSet {
  std::string name;
  std::string folder;
  double factor;
};

class DQMScaleToClient: public DQMEDHarvester{

 public:

  DQMScaleToClient(const edm::ParameterSet& ps);
  virtual ~DQMScaleToClient() = default;
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

//private variables

  //variables from config file
  // Histograms
  MonitorElement* scaled_;

  MEPSet inputmepset_;
  OutputMEPSet outputmepset_;

};


#endif // DQMSCALETOCLIENT_H
