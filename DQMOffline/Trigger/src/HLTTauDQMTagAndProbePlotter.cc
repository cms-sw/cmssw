#include "DQMOffline/Trigger/interface/HLTTauDQMTagAndProbePlotter.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

#include <boost/algorithm/string.hpp>

namespace {
  std::string stripVersion(const std::string& pathName) {
    size_t versionStart = pathName.rfind("_v");
    if(versionStart == std::string::npos)
      return pathName;
    return pathName.substr(0, versionStart);
  }
}

HLTTauDQMTagAndProbePlotter::HLTTauDQMTagAndProbePlotter(const edm::ParameterSet& iConfig, GenericTriggerEventFlag* numFlag, GenericTriggerEventFlag* denFlag, const std::string& dqmBaseFolder) :
  HLTTauDQMPlotter(stripVersion(iConfig.getParameter<std::string>("name")), dqmBaseFolder),
  nbins_(iConfig.getParameter<int>("nbins")),
  xmin_(iConfig.getParameter<double>("xmin")),
  xmax_(iConfig.getParameter<double>("xmax")),
  xvariable(iConfig.getParameter<std::string>("xvariable"))
{
  num_genTriggerEventFlag_ = numFlag;
  den_genTriggerEventFlag_ = denFlag;

  boost::algorithm::to_lower(xvariable);
}

#include <algorithm>
void HLTTauDQMTagAndProbePlotter::bookHistograms(DQMStore::IBooker &iBooker,edm::Run const &iRun, edm::EventSetup const &iSetup) {
  if(!isValid())
    return;

  // Initialize the GenericTriggerEventFlag
  if ( num_genTriggerEventFlag_ && num_genTriggerEventFlag_->on() ) num_genTriggerEventFlag_->initRun( iRun, iSetup );
  if ( den_genTriggerEventFlag_ && den_genTriggerEventFlag_->on() ) den_genTriggerEventFlag_->initRun( iRun, iSetup );

  // Efficiency helpers
  iBooker.setCurrentFolder(triggerTag()+"/helpers");
  h_num = iBooker.book1D(xvariable+"EtEffNum",    "", nbins_, xmin_, xmax_);
  h_den = iBooker.book1D(xvariable+"EtEffDenom",    "", nbins_, xmin_, xmax_);
  iBooker.setCurrentFolder(triggerTag());
}


HLTTauDQMTagAndProbePlotter::~HLTTauDQMTagAndProbePlotter() {
  if (num_genTriggerEventFlag_) delete num_genTriggerEventFlag_;
  if (den_genTriggerEventFlag_) delete den_genTriggerEventFlag_;
}

void HLTTauDQMTagAndProbePlotter::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup, const HLTTauDQMOfflineObjects& refCollection) {

  std::vector<LV> offlineObjects;
  if(xvariable == "tau")      offlineObjects = refCollection.taus;
  if(xvariable == "muon")     offlineObjects = refCollection.muons;
  if(xvariable == "electron") offlineObjects = refCollection.electrons;
  if(xvariable == "met")      offlineObjects = refCollection.met;

  for(const LV& offlineObject: offlineObjects) {
    double xvar = offlineObject.pt();

    // Filter out events if Trigger Filtering is requested
    if (den_genTriggerEventFlag_->on() && ! den_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;

    h_den->Fill(xvar);


    // applying selection for numerator
    if (num_genTriggerEventFlag_->on() && ! num_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;

    h_num->Fill(xvar);
  }
}
