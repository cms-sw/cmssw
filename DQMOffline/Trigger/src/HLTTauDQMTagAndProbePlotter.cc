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

HLTTauDQMTagAndProbePlotter::HLTTauDQMTagAndProbePlotter(const edm::ParameterSet& iConfig, std::unique_ptr<GenericTriggerEventFlag> numFlag, std::unique_ptr<GenericTriggerEventFlag> denFlag, const std::string& dqmBaseFolder) :
  HLTTauDQMPlotter(stripVersion(iConfig.getParameter<std::string>("name")), dqmBaseFolder),
  nbinsPt_(iConfig.getParameter<int>("nPtBins")),
  ptmin_(iConfig.getParameter<double>("ptmin")),
  ptmax_(iConfig.getParameter<double>("ptmax")),
//  nbinsEta_(iConfig.getParameter<int>("nEtaBins")),
//  etamin_(iConfig.getParameter<double>("etamin")),
//  etamax_(iConfig.getParameter<double>("etamax")),
  nbinsPhi_(iConfig.getParameter<int>("nPhiBins")),
  phimin_(iConfig.getParameter<double>("phimin")),
  phimax_(iConfig.getParameter<double>("phimax")),
  xvariable(iConfig.getParameter<std::string>("xvariable"))
{
  num_genTriggerEventFlag_ = std::move(numFlag);
  den_genTriggerEventFlag_ = std::move(denFlag);

  boost::algorithm::to_lower(xvariable);

  if(xvariable!="met"){
    nbinsEta_ = iConfig.getParameter<int>("nEtaBins");
    etamin_   = iConfig.getParameter<double>("etamin");
    etamax_   = iConfig.getParameter<double>("etamax");
  }

  nOfflineObjs = iConfig.getUntrackedParameter<unsigned int>("nOfflObjs",1);
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
  h_num_pt = iBooker.book1D(xvariable+"EtEffNum",    "", nbinsPt_, ptmin_, ptmax_);
  h_den_pt = iBooker.book1D(xvariable+"EtEffDenom",    "", nbinsPt_, ptmin_, ptmax_);

  if(xvariable != "met"){
  h_num_eta = iBooker.book1D(xvariable+"EtaEffNum",    "", nbinsEta_, etamin_, etamax_);
  h_den_eta = iBooker.book1D(xvariable+"EtaEffDenom",    "", nbinsEta_, etamin_, etamax_);

  h_num_etaphi = iBooker.book2D(xvariable+"EtaPhiEffNum",    "", nbinsEta_, etamin_, etamax_, nbinsPhi_, phimin_, phimax_);
  h_den_etaphi = iBooker.book2D(xvariable+"EtaPhiEffDenom",    "", nbinsEta_, etamin_, etamax_, nbinsPhi_, phimin_, phimax_);
  h_den_etaphi->getTH2F()->SetOption("COL");                 
  }

  h_num_phi = iBooker.book1D(xvariable+"PhiEffNum",    "", nbinsPhi_, phimin_, phimax_);
  h_den_phi = iBooker.book1D(xvariable+"PhiEffDenom",    "", nbinsPhi_, phimin_, phimax_);

  iBooker.setCurrentFolder(triggerTag());
}


HLTTauDQMTagAndProbePlotter::~HLTTauDQMTagAndProbePlotter() = default;

void HLTTauDQMTagAndProbePlotter::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup, const HLTTauDQMOfflineObjects& refCollection) {

  std::vector<LV> offlineObjects;
  if(xvariable == "tau")      offlineObjects = refCollection.taus;
  if(xvariable == "muon")     offlineObjects = refCollection.muons;
  if(xvariable == "electron") offlineObjects = refCollection.electrons;
  if(xvariable == "met")      offlineObjects = refCollection.met;

  if(offlineObjects.size() < nOfflineObjs) return;

  for(const LV& offlineObject: offlineObjects) {

    // Filter out events if Trigger Filtering is requested
    if (den_genTriggerEventFlag_->on() && ! den_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;

    h_den_pt->Fill(offlineObject.pt());
    if(xvariable != "met"){
      h_den_eta->Fill(offlineObject.eta());
      h_den_etaphi->Fill(offlineObject.eta(),offlineObject.phi());
    }
    h_den_phi->Fill(offlineObject.phi());


    // applying selection for numerator
    if (num_genTriggerEventFlag_->on() && ! num_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;

    h_num_pt->Fill(offlineObject.pt());
    if(xvariable != "met"){
      h_num_eta->Fill(offlineObject.eta());
      h_num_etaphi->Fill(offlineObject.eta(),offlineObject.phi());
    }
    h_num_phi->Fill(offlineObject.phi());


  }
}
