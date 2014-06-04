// -*- C++ -*-
//
// Package:    OverlapProblemTSOSPositionFilter
// Class:      OverlapProblemTSOSPositionFilter
// 
/**\class OverlapProblemTSOSPositionFilter OverlapProblemTSOSPositionFilter.cc DebugTools/OverlapProblem/plugins/OverlapProblemTSOSPositionFilter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Venturi
//         Created:  Thu Dec 16 16:32:56 CEST 2010
// $Id: OverlapProblemTSOSPositionFilter.cc,v 1.1 2012/03/12 14:46:20 venturia Exp $
//
//


// system include files
#include <memory>
#include <numeric>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#include "FWCore/ServiceRegistry/interface/Service.h"
//#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "TH1F.h"
//
// class decleration
//


class OverlapProblemTSOSPositionFilter : public edm::EDFilter {
public:
  explicit OverlapProblemTSOSPositionFilter(const edm::ParameterSet&);
  ~OverlapProblemTSOSPositionFilter();
  
private:
  virtual void beginJob() ;
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
      // ----------member data ---------------------------

  const bool m_validOnly; 
  edm::InputTag m_ttacollection;

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
OverlapProblemTSOSPositionFilter::OverlapProblemTSOSPositionFilter(const edm::ParameterSet& iConfig):
  m_validOnly(iConfig.getParameter<bool>("onlyValidRecHit")),
  m_ttacollection(iConfig.getParameter<edm::InputTag>("trajTrackAssoCollection"))

{
   //now do what ever initialization is needed



}


OverlapProblemTSOSPositionFilter::~OverlapProblemTSOSPositionFilter()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
bool
OverlapProblemTSOSPositionFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  // loop on trajectories and plot TSOS local coordinate
  
  TrajectoryStateCombiner tsoscomb;
  
  // Trajectory Handle
  
  Handle<TrajTrackAssociationCollection> ttac;
  iEvent.getByLabel(m_ttacollection,ttac);
  
  for(TrajTrackAssociationCollection::const_iterator pair=ttac->begin();pair!=ttac->end();++pair) {
    
    const edm::Ref<std::vector<Trajectory> > & traj = pair->key;
    //    const reco::TrackRef & trk = pair->val;
    const std::vector<TrajectoryMeasurement> & tmcoll = traj->measurements();
    
    for(std::vector<TrajectoryMeasurement>::const_iterator meas = tmcoll.begin() ; meas!= tmcoll.end() ; ++meas) {
      
      if(!meas->updatedState().isValid()) continue;
      
      TrajectoryStateOnSurface tsos = tsoscomb(meas->forwardPredictedState(), meas->backwardPredictedState());
      TransientTrackingRecHit::ConstRecHitPointer hit = meas->recHit();
      
      if(!hit->isValid() && m_validOnly) continue;

      if(hit->geographicalId().det() != DetId::Tracker) continue;
      
      TECDetId det(hit->geographicalId());
      if(det.subDetector() != SiStripDetId::TEC) continue;
      
      if(det.ring() != 6) continue;

      if(tsos.localPosition().y() < 6.) continue; 

      return true;
      
    }
    
  }
  
  return false;

}

// ------------ method called once each job just before starting event loop  ------------
void 
OverlapProblemTSOSPositionFilter::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
OverlapProblemTSOSPositionFilter::endJob() 
{
}

//define this as a plug-in
DEFINE_FWK_MODULE(OverlapProblemTSOSPositionFilter);
