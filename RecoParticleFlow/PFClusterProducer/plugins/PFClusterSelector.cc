#include "RecoParticleFlow/PFClusterProducer/plugins/PFClusterSelector.h"
#include <memory>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"


using namespace std;
using namespace edm;


PFClusterSelector::PFClusterSelector(const edm::ParameterSet& iConfig):
  clusters_(consumes<reco::PFClusterCollection>(iConfig.getParameter<edm::InputTag>("src"))),
  energyRanges_(iConfig.getParameter<std::vector<double> >("energyRanges")),
  timingCutsLow_(iConfig.getParameter<std::vector<double> >("timingCutsLow")),
  timingCutsHigh_(iConfig.getParameter<std::vector<double> >("timingCutsHigh")),
  timingCutsLowEE_(iConfig.getParameter<std::vector<double> >("timingCutsEndcapLow")),
  timingCutsHighEE_(iConfig.getParameter<std::vector<double> >("timingCutsEndcapHigh"))
{
  produces<reco::PFClusterCollection>();
}


void PFClusterSelector::produce(edm::Event& iEvent, 
			       const edm::EventSetup& iSetup) {

  edm::Handle<reco::PFClusterCollection> clusters; 
  iEvent.getByToken(clusters_,clusters);
  std::auto_ptr<reco::PFClusterCollection> out(new reco::PFClusterCollection);

  for(unsigned int i=0;i<clusters->size();++i) {
    double energy = clusters->at(i).energy();
    double time = clusters->at(i).time();

    if (clusters->at(i).layer() == PFLayer::ECAL_BARREL ||
	clusters->at(i).layer() == PFLayer::HCAL_BARREL1 ||
	clusters->at(i).layer() == PFLayer::HCAL_BARREL2) {

      for(unsigned int j=0;j<energyRanges_.size();++j) {
	if (j==0 && energy<energyRanges_.at(0)) {
	  if ( time >timingCutsLow_[j] &&  time <timingCutsHigh_[j] )
	    out->push_back(clusters->at(i));
	}
	else if (j==clusters->size()-1 && energy>=energyRanges_.at(j)) {
	  if ( time >timingCutsLow_[j] &&  time <timingCutsHigh_[j] )
	    out->push_back(clusters->at(i));
	}
	else {
	  if ( energy>energyRanges_[j-1] && energy < energyRanges_[j]&& time >timingCutsLow_[j] &&  time <timingCutsHigh_[j] )
	    out->push_back(clusters->at(i));
	}
      }
	
    }
    else 
      {
      for(unsigned int j=0;j<energyRanges_.size();++j) {
	if (j==0 && energy<energyRanges_.at(0)) {
	  if ( time >timingCutsLowEE_[j] &&  time <timingCutsHighEE_[j] )
	    out->push_back(clusters->at(i));
	}
	else if (j==clusters->size()-1 && energy>=energyRanges_.at(j)) {
	  if ( time >timingCutsLowEE_[j] &&  time <timingCutsHighEE_[j] )
	    out->push_back(clusters->at(i));
	}
	else {
	  if ( energy>energyRanges_[j-1] && energy < energyRanges_[j]&& time >timingCutsLowEE_[j] &&  time <timingCutsHighEE_[j] )
	    out->push_back(clusters->at(i));
	}
      }
	


      }

  }
  iEvent.put( out);

}

PFClusterSelector::~PFClusterSelector() {}

// ------------ method called once each job just before starting event loop  ------------
void 
PFClusterSelector::beginRun(const edm::Run& run,
			   const EventSetup& es) {

 
}


