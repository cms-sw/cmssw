#include "RecoTauTag/InvariantMass/interface/InvariantMassAlgorithm.h"
#include <boost/regex.hpp>

using namespace std; 
using namespace edm;
using namespace reco;

//
// -- Constructor
//
InvariantMassAlgorithm::InvariantMassAlgorithm()
{
  matching_cone               = 0.1; // check with simone
  leading_trk_pt              = 1.0; // check with simone
  signal_cone                 = 0.07;// check with simone
  cluster_jet_matching_cone   = 0.4;
  cluster_track_matching_cone = 0.08;
  inv_mass_cut                = 2.5;
}
//
// -- Constructor
//
InvariantMassAlgorithm::InvariantMassAlgorithm(const ParameterSet& parameters)
{
  matching_cone               = parameters.getParameter<double>("MatchingCone");
  leading_trk_pt              = parameters.getParameter<double>("LeadingTrackPt");
  signal_cone                 = parameters.getParameter<double>("SignalCone");
  cluster_jet_matching_cone   = parameters.getParameter<double>("ClusterSelectionCone");
  cluster_track_matching_cone = parameters.getParameter<double>("ClusterTrackMatchingCone");
  inv_mass_cut                = parameters.getParameter<double>("InvariantMassCutoff");


// TrackAssociator parameters
   edm::ParameterSet tk_ass_pset = parameters.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
   trackAssociatorParameters_.loadParameters( tk_ass_pset );

   trackAssociator_ = new TrackDetectorAssociator();
   trackAssociator_->useDefaultPropagator();
}
//
// -- Destructor
//
InvariantMassAlgorithm::~InvariantMassAlgorithm() {
  if(trackAssociator_) delete trackAssociator_;
}
//
// -- Tag 
//
pair<double, reco::TauMassTagInfo> InvariantMassAlgorithm::tag(edm::Event& theEvent, 
                                                              const edm::EventSetup& theEventSetup, 
                                                              const reco::IsolatedTauTagInfoRef& tauRef, 
                                                              const Handle<BasicClusterCollection>& clus_handle) 
{

  TauMassTagInfo resultExtended;
  resultExtended.setIsolatedTauTag(tauRef);

  double discriminator = tauRef->discriminator();
  if (discriminator > 0.0) {

    int nSel = 0;
    for (size_t ic = 0; ic < clus_handle->size(); ic++) {
      math::XYZVector cluster3Vec((*clus_handle)[ic].x(),(*clus_handle)[ic].y(),(*clus_handle)[ic].z());
      BasicClusterRef cluster(clus_handle, ic);
      float et = (*clus_handle)[ic].energy() * sin(cluster3Vec.theta());       
      if ( (et < 0.04) || (et > 300.0) ) continue;    
      float delR = getMinimumClusterDR(theEvent, theEventSetup, tauRef, cluster3Vec);
      if (delR != -1.0) {
        nSel++;
        resultExtended.storeClusterTrackCollection(cluster, delR);
      }
    }
    if (0) cout << " Total Number of Clusters " << clus_handle->size() << " " << nSel << endl;
    
    discriminator = resultExtended.discriminator(matching_cone,
                                                 leading_trk_pt,
                                                 signal_cone,
                                                 cluster_track_matching_cone,
                                                 inv_mass_cut); 
  }
  return make_pair(discriminator, resultExtended); 
}
//
// get Cluster Map
//
float InvariantMassAlgorithm::getMinimumClusterDR(edm::Event& theEvent,  
                                         const edm::EventSetup& theEventSetup, 
                                         const reco::IsolatedTauTagInfoRef& tauRef,
                                         const math::XYZVector& cluster_3vec) 
{

  trackAssociatorParameters_.useEcal = true;
  trackAssociatorParameters_.useHcal = false;
  trackAssociatorParameters_.useHO   = false;
  trackAssociatorParameters_.useMuon = false;
  trackAssociatorParameters_.dREcal  = 0.03;

  const TrackRefVector tracks = tauRef->allTracks();
  const Jet & jet = *(tauRef->jet()); 
  math::XYZVector jet3Vec(jet.px(),jet.py(),jet.pz());   
  float min_dR = 999.9; 
  float deltaR1 = ROOT::Math::VectorUtil::DeltaR(cluster_3vec, jet3Vec);
  if (deltaR1 > cluster_jet_matching_cone) return -1.0;

  for (edm::RefVector<reco::TrackCollection>::const_iterator it  = tracks.begin();
                                                             it != tracks.end(); it++) 
  {    
    TrackDetMatchInfo info;
    info = trackAssociator_->associate(theEvent, theEventSetup,
                                     trackAssociator_->getFreeTrajectoryState(theEventSetup, (*(*it))), 
                                     trackAssociatorParameters_);
    math::XYZVector track3Vec(info.trkGlobPosAtEcal.x(),
                              info.trkGlobPosAtEcal.y(),
                              info.trkGlobPosAtEcal.z());

    float deltaR2 = ROOT::Math::VectorUtil::DeltaR(cluster_3vec, track3Vec);
    if (deltaR2 < min_dR) min_dR = deltaR2;        
  }
  return min_dR;
}
