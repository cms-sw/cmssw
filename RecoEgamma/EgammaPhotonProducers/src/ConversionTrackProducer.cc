//
// Package:         RecoTracker/FinalTrackSelectors
// Class:           ConversionTrackProducer
// 
// Description:     Trivial producer of ConversionTrack collection from an edm::View of a track collection
//                  (ConversionTrack is a simple wrappper class containing a TrackBaseRef and some additional flags)
//
// Original Author: J.Bendavid
//
//

#include <memory>
#include <string>
#include <iostream>
#include <cmath>
#include <vector>

#include "TrackingTools/PatternTools/interface/Trajectory.h"

#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"

#include "RecoEgamma/EgammaPhotonProducers/interface/ConversionTrackProducer.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

    
ConversionTrackProducer::ConversionTrackProducer(edm::ParameterSet const& conf) : 
  conf_(conf),
  trackProducer ( conf.getParameter<std::string>("TrackProducer") ),
  useTrajectory ( conf.getParameter<bool>("useTrajectory") ),
  setTrackerOnly ( conf.getParameter<bool>("setTrackerOnly") ),
  setArbitratedEcalSeeded ( conf.getParameter<bool>("setArbitratedEcalSeeded") ),    
  setArbitratedMerged ( conf.getParameter<bool>("setArbitratedMerged") ),
  setArbitratedMergedEcalGeneral ( conf.getParameter<bool>("setArbitratedMergedEcalGeneral") ),
  beamSpotInputTag (  consumes<reco::BeamSpot>(conf.getParameter<edm::InputTag>("beamSpotInputTag")) ),
  filterOnConvTrackHyp( conf.getParameter<bool>("filterOnConvTrackHyp") ),
  minConvRadius( conf.getParameter<double>("minConvRadius") )
{
  edm::InputTag thetp(trackProducer);
  genericTracks =
    consumes<edm::View<reco::Track> >(thetp);
  kfTrajectories = 
    consumes<TrajTrackAssociationCollection>(thetp);
  gsfTrajectories = 
    consumes<TrajGsfTrackAssociationCollection>(thetp);
  produces<reco::ConversionTrackCollection>();
  
}


  // Virtual destructor needed.
  ConversionTrackProducer::~ConversionTrackProducer() { }  

  // Functions that gets called by framework every event
  void ConversionTrackProducer::produce(edm::Event& e, const edm::EventSetup& es)
  {
    //get input collection (through edm::View)
    edm::Handle<edm::View<reco::Track> > hTrks;
    e.getByToken(genericTracks, hTrks);

    //get association maps between trajectories and tracks and build temporary maps
    edm::Handle< TrajTrackAssociationCollection > hTTAss;
    edm::Handle< TrajGsfTrackAssociationCollection > hTTAssGsf;    
                                          
    std::map<reco::TrackRef,edm::Ref<std::vector<Trajectory> > > tracktrajmap;
    std::map<reco::GsfTrackRef,edm::Ref<std::vector<Trajectory> > > gsftracktrajmap;
                                          
    if (useTrajectory) {
      if (hTrks->size()>0) {
        if (dynamic_cast<const reco::GsfTrack*>(&hTrks->at(0))) {
          //fill map for gsf tracks
          e.getByToken(gsfTrajectories, hTTAssGsf);     
          for ( TrajGsfTrackAssociationCollection::const_iterator iPair = 
		  hTTAssGsf->begin();
		iPair != hTTAssGsf->end(); ++iPair) {
        
            gsftracktrajmap[iPair->val] = iPair->key;

          }
                
        }
        else {
          //fill map for standard tracks
          e.getByToken(kfTrajectories, hTTAss);
          for ( TrajTrackAssociationCollection::const_iterator iPair = hTTAss->begin();
            iPair != hTTAss->end();
            ++iPair) {
        
            tracktrajmap[iPair->val] = iPair->key;

          }
        }
      }
    }

    // Step B: create empty output collection
    outputTrks = std::auto_ptr<reco::ConversionTrackCollection>(new reco::ConversionTrackCollection);    

    //--------------------------------------------------
    //Added by D. Giordano
    // 2011/08/05
    // Reduction of the track sample based on geometric hypothesis for conversion tracks
 
    edm::Handle<reco::BeamSpot> beamSpotHandle;
    e.getByToken(beamSpotInputTag,beamSpotHandle);
   
    edm::ESHandle<MagneticField> magFieldHandle;
    es.get<IdealMagneticFieldRecord>().get( magFieldHandle );


    if(filterOnConvTrackHyp && !beamSpotHandle.isValid()) {
      edm::LogError("Invalid Collection") 
	<< "invalid collection for the BeamSpot";
      throw;
    }

    ConvTrackPreSelector.setMagnField(magFieldHandle.product());

    //----------------------------------------------------------
   
 
    // Simple conversion of tracks to conversion tracks, setting appropriate flags from configuration
    for (size_t i = 0; i < hTrks->size(); ++i) {
 
      //--------------------------------------------------
      //Added by D. Giordano
      // 2011/08/05
      // Reduction of the track sample based on geometric hypothesis for conversion tracks
      
      math::XYZVector beamSpot=  math::XYZVector(beamSpotHandle->position());
      edm::RefToBase<reco::Track> trackBaseRef = hTrks->refAt(i);
      if( filterOnConvTrackHyp && ConvTrackPreSelector.isTangentPointDistanceLessThan( minConvRadius, trackBaseRef.get(), beamSpot )  )
	continue;
      //--------------------------------------------------

      reco::ConversionTrack convTrack(trackBaseRef);
      convTrack.setIsTrackerOnly(setTrackerOnly);
      convTrack.setIsArbitratedEcalSeeded(setArbitratedEcalSeeded);
      convTrack.setIsArbitratedMerged(setArbitratedMerged);
      convTrack.setIsArbitratedMergedEcalGeneral(setArbitratedMergedEcalGeneral);
            
      //fill trajectory association if configured, using correct map depending on track type
      if (useTrajectory) {
        if (gsftracktrajmap.size()) {
          convTrack.setTrajRef(gsftracktrajmap.find(trackBaseRef.castTo<reco::GsfTrackRef>())->second);
        }
        else {
          convTrack.setTrajRef(tracktrajmap.find(trackBaseRef.castTo<reco::TrackRef>())->second);
        }
      }
      
      outputTrks->push_back(convTrack);
    }
    
    e.put(outputTrks);
    return;

  }//end produce

