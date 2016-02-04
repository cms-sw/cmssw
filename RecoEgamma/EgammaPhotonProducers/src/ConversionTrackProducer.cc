//
// Package:         RecoTracker/FinalTrackSelectors
// Class:           ConversionTrackProducer
// 
// Description:     Trivial producer of ConversionTrack collection from an edm::View of a track collection
//                  (ConversionTrack is a simple wrappper class containing a TrackBaseRef and some additional flags)
//
// Original Author: J.Bendavid
//
// $Author: giordano $
// $Date: 2011/08/05 19:45:49 $
// $Revision: 1.4 $
//

#include <memory>
#include <string>
#include <iostream>
#include <cmath>
#include <vector>

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "TrackingTools/GsfTracking/interface/TrajGsfTrackAssociation.h"
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
  beamSpotInputTag (  conf.getParameter<edm::InputTag>("beamSpotInputTag") ),
  filterOnConvTrackHyp( conf.getParameter<bool>("filterOnConvTrackHyp") ),
  minConvRadius( conf.getParameter<double>("minConvRadius") )
{

  produces<reco::ConversionTrackCollection>();
  
}


  // Virtual destructor needed.
  ConversionTrackProducer::~ConversionTrackProducer() { }  

  // Functions that gets called by framework every event
  void ConversionTrackProducer::produce(edm::Event& e, const edm::EventSetup& es)
  {
    //get input collection (through edm::View)
    edm::Handle<edm::View<reco::Track> > hTrks;
    e.getByLabel(trackProducer, hTrks);

    //get association maps between trajectories and tracks and build temporary maps
    edm::Handle< TrajTrackAssociationCollection > hTTAss;
    edm::Handle< edm::AssociationMap<edm::OneToOne<std::vector<Trajectory>,
                                          reco::GsfTrackCollection,unsigned short> > > hTTAssGsf;    
                                          
    std::map<reco::TrackRef,edm::Ref<std::vector<Trajectory> > > tracktrajmap;
    std::map<reco::GsfTrackRef,edm::Ref<std::vector<Trajectory> > > gsftracktrajmap;
                                          
    if (useTrajectory) {
      if (hTrks->size()>0) {
        if (dynamic_cast<const reco::GsfTrack*>(&hTrks->at(0))) {
          //fill map for gsf tracks
          e.getByLabel(trackProducer, hTTAssGsf);     
          for ( edm::AssociationMap<edm::OneToOne<std::vector<Trajectory>,
                    reco::GsfTrackCollection,unsigned short> >::const_iterator iPair = hTTAssGsf->begin();
            iPair != hTTAssGsf->end(); ++iPair) {
        
            gsftracktrajmap[iPair->val] = iPair->key;

          }
                
        }
        else {
          //fill map for standard tracks
          e.getByLabel(trackProducer, hTTAss);
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
    e.getByLabel(beamSpotInputTag,beamSpotHandle);
   
    edm::ESHandle<MagneticField> magFieldHandle;
    es.get<IdealMagneticFieldRecord>().get( magFieldHandle );


    if(filterOnConvTrackHyp && !beamSpotHandle.isValid()) {
      edm::LogError("Invalid Collection") << "invalid collection for the BeamSpot with InputTag " << beamSpotInputTag;
      throw;
    }

    ConvTrackPreSelector.setMagnField(magFieldHandle.product());

    //----------------------------------------------------------
   
 
    // Simple conversion of tracks to conversion tracks, setting appropriate flags from configuration
    for (edm::RefToBaseVector<reco::Track>::const_iterator it = hTrks->refVector().begin(); it != hTrks->refVector().end(); ++it) {
 
      //--------------------------------------------------
      //Added by D. Giordano
      // 2011/08/05
      // Reduction of the track sample based on geometric hypothesis for conversion tracks
      
      math::XYZVector beamSpot=  math::XYZVector(beamSpotHandle->position());

      if( filterOnConvTrackHyp && ConvTrackPreSelector.isTangentPointDistanceLessThan( minConvRadius, it->get(), beamSpot )  )
	continue;
      //--------------------------------------------------

      reco::ConversionTrack convTrack(*it);
      convTrack.setIsTrackerOnly(setTrackerOnly);
      convTrack.setIsArbitratedEcalSeeded(setArbitratedEcalSeeded);
      convTrack.setIsArbitratedMerged(setArbitratedMerged);
      convTrack.setIsArbitratedMergedEcalGeneral(setArbitratedMergedEcalGeneral);
            
      //fill trajectory association if configured, using correct map depending on track type
      if (useTrajectory) {
        if (gsftracktrajmap.size()) {
          convTrack.setTrajRef(gsftracktrajmap.find(it->castTo<reco::GsfTrackRef>())->second);
        }
        else {
          convTrack.setTrajRef(tracktrajmap.find(it->castTo<reco::TrackRef>())->second);
        }
      }
      
      outputTrks->push_back(convTrack);
    }
    
    e.put(outputTrks);
    return;

  }//end produce

