// -*- C++ -*-
//
// Package:    ConfigurableAnalysis
// Class:      ConfigurableAnalysis
// 
/**\class ConfigurableAnalysis ConfigurableAnalysis.cc CommonTools/UtilAlgos/src/ConfigurableAnalysis.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jean-Roch Vlimant
//         Created:  Mon Apr 14 11:39:51 CEST 2008
// $Id: ConfigurableAnalysis.cc,v 1.10 2010/09/28 09:08:33 srappocc Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/UtilAlgos/interface/Selections.h"
#include "PhysicsTools/UtilAlgos/interface/Plotter.h"
#include "PhysicsTools/UtilAlgos/interface/NTupler.h"
#include "PhysicsTools/UtilAlgos/interface/InputTagDistributor.h"

//
// class decleration
//

class ConfigurableAnalysis : public edm::EDFilter {
   public:
      explicit ConfigurableAnalysis(const edm::ParameterSet&);
      ~ConfigurableAnalysis();

   private:
      virtual void beginJob();
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

  Selections * selections_;
  Plotter * plotter_;
  NTupler * ntupler_;

  std::vector<std::string> flows_;
  bool workAsASelector_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
ConfigurableAnalysis::ConfigurableAnalysis(const edm::ParameterSet& iConfig) :
  selections_(0), plotter_(0), ntupler_(0)
{

  std::string moduleLabel = iConfig.getParameter<std::string>("@module_label");

  //configure inputag distributor
  if (iConfig.exists("InputTags"))
    edm::Service<InputTagDistributorService>()->init(moduleLabel,iConfig.getParameter<edm::ParameterSet>("InputTags"));

  //configure the variable helper
  edm::Service<VariableHelperService>()->init(moduleLabel,iConfig.getParameter<edm::ParameterSet>("Variables"));

  //list of selections
  selections_ = new Selections(iConfig.getParameter<edm::ParameterSet>("Selections"));

  //plotting device
  edm::ParameterSet plotPset = iConfig.getParameter<edm::ParameterSet>("Plotter");
  if (!plotPset.empty()){
    std::string plotterName = plotPset.getParameter<std::string>("ComponentName");
    plotter_ = PlotterFactory::get()->create(plotterName, plotPset);
  }
  else
    plotter_ = 0;

  //ntupling device
  edm::ParameterSet ntPset = iConfig.getParameter<edm::ParameterSet>("Ntupler");
  if (!ntPset.empty()){
    std::string ntuplerName=ntPset.getParameter<std::string>("ComponentName");
    ntupler_ = NTuplerFactory::get()->create(ntuplerName, ntPset);
  }
  else ntupler_=0;
  
  flows_ = iConfig.getParameter<std::vector<std::string> >("flows");
  workAsASelector_ = iConfig.getParameter<bool>("workAsASelector");

  //vector of passed selections
  produces<std::vector<bool> >();

  //ntupler needs to register its products
  if (ntupler_) ntupler_->registerleaves(this);
}

ConfigurableAnalysis::~ConfigurableAnalysis()
{
  delete selections_;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
bool ConfigurableAnalysis::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  //will the filter pass or not.
  bool majorGlobalAccept=false;

  std::auto_ptr<std::vector<bool> > passedProduct(new std::vector<bool>(flows_.size(),false));
  bool filledOnce=false;  

  // loop the requested selections
  for (Selections::iterator selection=selections_->begin(); selection!=selections_->end();++selection){
    //was this flow of filter actually asked for
    bool skip=true;
    unsigned int iFlow=0;
    for (;iFlow!=flows_.size();++iFlow){if (flows_[iFlow]==selection->name()){skip=false; break;}}
    if (skip) continue;

    //make a specific direction in the plotter
    if (plotter_) plotter_->setDir(selection->name());
    
    // apply individual filters on the event
    std::map<std::string, bool> accept=selection->accept(iEvent);
    
    bool globalAccept=true;
    std::string separator="";
    std::string cumulative="";
    std::string allButOne="allBut_";
    std::string fullAccept="fullAccept";
    std::string fullContent="fullContent";

    if (selection->makeContentPlots() && plotter_)
      plotter_->fill(fullContent,iEvent);

    //loop the filters to make cumulative and allButOne job
    for (Selection::iterator filterIt=selection->begin(); filterIt!=selection->end();++filterIt){
      Filter & filter=(**filterIt);
      //      bool lastCut=((filterIt+1)==selection->end());

      //increment the directory name
      cumulative+=separator+filter.name(); separator="_";

      if (accept[filter.name()]){
	//	if (globalAccept && selection->makeCumulativePlots() && !lastCut)
	if (globalAccept && selection->makeCumulativePlots() && plotter_)
	  plotter_->fill(cumulative,iEvent);
      }
      else{
	globalAccept=false;
	// did all the others filter fire
	bool goodForAllButThisOne=true;
	for (std::map<std::string,bool>::iterator decision=accept.begin(); decision!=accept.end();++decision){
	  if (decision->first==filter.name()) continue;
	  if (!decision->second) {
	    goodForAllButThisOne=false;
	    break;}
	}
	if (goodForAllButThisOne && selection->makeAllButOnePlots() && plotter_){
	  plotter_->fill(allButOne+filter.name(),iEvent);
	}
      }
      
    }// loop over the filters in this selection

    if (globalAccept){
      (*passedProduct)[iFlow]=true;
      majorGlobalAccept=true;
      //make final plots only if no cumulative plots
      if (selection->makeFinalPlots() && !selection->makeCumulativePlots() && plotter_)
	plotter_->fill(fullAccept,iEvent);

      //make the ntuple and put it in the event
      if (selection->ntuplize() && !filledOnce && ntupler_){
	ntupler_->fill(iEvent);
	filledOnce=true;}
    }
    
  }//loop the different filter order/number: loop the Selections

  iEvent.put(passedProduct);
  if (workAsASelector_)
    return majorGlobalAccept;
  else
    return true;
}
   

// ------------ method called once each job just before starting event loop  ------------
void 
ConfigurableAnalysis::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
ConfigurableAnalysis::endJob() {
  //print summary tables
  selections_->print();
  if (plotter_) plotter_->complete();
}


DEFINE_FWK_MODULE(ConfigurableAnalysis);
