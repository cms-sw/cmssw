/** \class DTandCSCSegmentsinTracks
 *
 *  Producer which take as input a muon track and return two containers 
 *  with the DTSegments and CSCSegments (respectively) used to fit it
 *
 *  $Date: 2012/01/19 16:01:03 $
 *  
 *  \author Juan Pablo Gomez - Uniandes
 */
#include <vector>

//standard include
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h" 
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
///

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDProducer.h"
//#include<fstream>






class DTandCSCSegmentsinTracks : public edm::EDProducer {
   public:
      explicit DTandCSCSegmentsinTracks(const edm::ParameterSet&);
      ~DTandCSCSegmentsinTracks();
     
   private:
      edm::InputTag cscSegments;
      edm::InputTag dt4DSegments;
      edm::InputTag tracks;
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      // ----------member data ---------------------------
};

