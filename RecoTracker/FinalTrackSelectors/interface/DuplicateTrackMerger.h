#ifndef RecoTracker_DuplicateTrackMerger_h
#define RecoTracker_DuplicateTrackMerger_h
/** \class DuplicateTrackMerger
 * 
 * selects pairs of tracks that should be single tracks
 *
 * \author Matthew Walker
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "RecoTracker/TrackProducer/interface/TrackMerger.h"
#include <vector>
#include <algorithm>
#include <string>
#include <iostream>
#include <map>

#include "TMVA/Reader.h"

namespace reco { namespace modules {
    class DuplicateTrackMerger : public edm::EDProducer {
       public:
         /// constructor
         explicit DuplicateTrackMerger(const edm::ParameterSet& iPara);
	 /// destructor
	 virtual ~DuplicateTrackMerger();

	 typedef std::vector<std::pair<TrackCandidate,std::pair<reco::Track,reco::Track> > > CandidateToDuplicate;

       protected:
	 /// produce one event
	 void produce( edm::Event &, const edm::EventSetup &);

       private:
	 /// MVA discriminator
	 TMVA::Reader* tmvaReader_;
	 /// MVA input variables
	 float tmva_ddsz_;
	 float tmva_ddxy_;
	 float tmva_dphi_;
	 float tmva_dlambda_;
	 float tmva_dqoverp_;
	 float tmva_d3dr_;
	 float tmva_d3dz_;
	 float tmva_outer_nMissingInner_;
	 float tmva_inner_nMissingOuter_;

	 /// track input collection
	 edm::InputTag trackSource_;
	 /// MVA weights file
	 std::string weightFileName_;
	 /// minDeltaR3d cut value
	 double minDeltaR3d_;
	 /// minBDTG cut value
	 double minBDTG_;


	 edm::ESHandle<MagneticField> magfield_;

	 ///Merger
	 TrackMerger merger_;
     };
  }
}
#endif
