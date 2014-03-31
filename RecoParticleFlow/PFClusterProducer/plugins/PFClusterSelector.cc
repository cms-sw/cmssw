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

  const std::vector<double>* timingCutsLow = NULL;
  const std::vector<double>* timingCutsHigh = NULL;
  for(const auto& cluster : *clusters ) {    
    const double energy = cluster.energy();
    const double time = cluster.time();    

    switch(cluster.layer()) {
    case PFLayer::ECAL_BARREL:
    case PFLayer::HCAL_BARREL1:
    case PFLayer::HCAL_BARREL2:
      timingCutsLow = &timingCutsLowBarrel_;
      timingCutsHigh = &timingCutsHighBarrel_;
      break;
    case PFLayer::ECAL_ENDCAP:
    case PFLayer::HCAL_ENDCAP:
      timingCutsLow = &timingCutsLowEndcap_;
      timingCutsHigh = &timingCutsHighEndcap_;
      break;
    default:      
      continue;
    }    
    auto e_bin = std::lower_bound(energyRanges_.begin(),energyRanges_.end(),energy);
    // Returns iterator to the first value that is *greater* than the value
    // we are comparing to. 
    // The timing cuts are indexed by the bin high edge so we just need the 
    // distance between this bin timing cut vectors are padded to avoid 
    // overflows.
    const unsigned idx = std::distance(energyRanges_.begin(),e_bin);
    if( time > (*timingCutsLow)[idx] && time < (*timingCutsHigh)[idx] ) {
      out->push_back(cluster);
    }
    else 
      {
	for(unsigned int j=0;j<energyRanges_.size();++j) {
	  if (j==0) { 
	    if ( energy<energyRanges_.at(0) && time >timingCutsLowEE_[j] &&  time <timingCutsHighEE_[j] )
	      out->push_back(clusters->at(i));
	  }
	  else if (j==(energyRanges_.size()-1) ) {
	    if ( time >timingCutsLowEE_[j] &&  time <timingCutsHighEE_[j] && energy>=energyRanges_.at(j) )
	      out->push_back(clusters->at(i));
	  }
	  else {
	    if ( energy>energyRanges_[j-1] && energy <= energyRanges_[j]&& time >timingCutsLowEE_[j] &&  time <timingCutsHighEE_[j] )
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


