#ifndef InvariantMassAlgorithm_H
#define InvariantMassAlgorithm_H

#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/BTauReco/interface/TauMassTagInfo.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

//Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"

class  InvariantMassAlgorithm  {

public:
 
  InvariantMassAlgorithm(const edm::ParameterSet  & parameters );
  InvariantMassAlgorithm(); 

  // For out of framework usage we may need a different constructor
  // so we keep datamember as builtin types (instead of ParameterSet) 
  //InvariantMassAlgorithm (int,float,....);
   
  ~InvariantMassAlgorithm();

  std::pair<double, reco::TauMassTagInfo> tag(edm::Event& theEvent, const edm::EventSetup& theEventSetup,const reco::IsolatedTauTagInfoRef& tauRef, const edm::Handle<reco::BasicClusterCollection>& clus_handle); 

  float getMinimumClusterDR(edm::Event& theEvent, const edm::EventSetup& theEventSetup,const reco::IsolatedTauTagInfoRef& tauRef, const math::XYZVector& cluster_3vec);

private:

//algorithm parameters
  
  double matching_cone;
  double leading_trk_pt;
  double signal_cone;
  double cluster_jet_matching_cone;
  double cluster_track_matching_cone;
  double inv_mass_cut;

  TrackDetectorAssociator* trackAssociator_;
  TrackAssociatorParameters trackAssociatorParameters_;

};

#endif 

