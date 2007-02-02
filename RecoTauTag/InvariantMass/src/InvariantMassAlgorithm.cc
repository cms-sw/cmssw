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
  isolation_cone = 0.1; // check with simone
  leading_trk_pt = 1.0; // check with simone
  signal_cone = 0.07;// check with simone
  jet_matching_cone = 0.4;
  track_matching_cone = 0.08;
  inv_mass_cut  = 2.5;
}
//
// -- Constructor
//
InvariantMassAlgorithm::InvariantMassAlgorithm(const ParameterSet & parameters)
{
  isolation_cone = parameters.getParameter<double>("IsolationCone");
  leading_trk_pt = parameters.getParameter<double>("LeadingTrackPt");
  signal_cone = parameters.getParameter<double>("SignalCone");
  jet_matching_cone = parameters.getParameter<double>("ClusterSelectionCone");
  track_matching_cone = parameters.getParameter<double>("ClusterTrackMatchingCone");
  inv_mass_cut  = parameters.getParameter<double>("InvariantMassCutoff");

   // Fill data labels
   trackAssociator_.theEBRecHitCollectionLabel = parameters.getParameter<edm::InputTag>("EBRecHitCollectionLabel");
   trackAssociator_.theEERecHitCollectionLabel = parameters.getParameter<edm::InputTag>("EERecHitCollectionLabel");
   trackAssociator_.theCaloTowerCollectionLabel = parameters.getParameter<edm::InputTag>("CaloTowerCollectionLabel");

   trackAssociator_.useDefaultPropagator();

}

//
// -- Tag 
//
pair<reco::JetTag,reco::TauMassTagInfo> InvariantMassAlgorithm::tag(edm::Event& theEvent, const edm::EventSetup& theEventSetup, const reco::IsolatedTauTagInfoRef& tauRef, const Handle<BasicClusterCollection>& clus_handle) {

  TauMassTagInfo resultExtended;
  resultExtended.setIsolatedTauTag(tauRef);
  int nSel = 0;
  for (size_t ic=0; ic < clus_handle->size() ; ic++) {
        
    math::XYZVector cluster3Vec((*clus_handle)[ic].x(),(*clus_handle)[ic].y(),(*clus_handle)[ic].z());
    BasicClusterRef cluster(clus_handle, ic);
    float delR = getMinimumClusterDR(theEvent, theEventSetup, tauRef,cluster3Vec);
    if (delR != -1.0) {
      nSel++;
      resultExtended.storeClusterTrackCollection(cluster,delR);
    }
  }
  if (0) cout << " Total Number of Clusters " << clus_handle->size() << " " << nSel << endl;

  double discriminator = tauRef->discriminator();
  if (discriminator > 0.0) {
    discriminator = resultExtended.discriminator(isolation_cone,leading_trk_pt,signal_cone,
                                         track_matching_cone,inv_mass_cut); 
  }
  const JetTracksAssociationRef& jtaRef = tauRef->jetRef()->jtaRef();
  JetTag resultBase(discriminator,jtaRef);

  return pair<JetTag,TauMassTagInfo> (resultBase,resultExtended); 
}
 //
// get  Cluster Map
//
float InvariantMassAlgorithm::getMinimumClusterDR(edm::Event& theEvent, const edm::EventSetup& theEventSetup, const reco::IsolatedTauTagInfoRef&  tauRef,const math::XYZVector& cluster_3vec) {


  TrackDetectorAssociator::AssociatorParameters assotiator_parameters;
  assotiator_parameters.useEcal = true ;
  assotiator_parameters.useHcal = false ;
  assotiator_parameters.useMuon = false ;
  assotiator_parameters.dREcal = 0.03;

  const TrackRefVector tracks = tauRef->allTracks();
  float min_dR = 999.9;
  math::XYZVector jet3Vec(tauRef->jet().px(),tauRef->jet().py(),tauRef->jet().pz());  
  float deltaR1 = ROOT::Math::VectorUtil::DeltaR(cluster_3vec, jet3Vec);
  if (deltaR1 > jet_matching_cone) return -1.0;

  for (edm::RefVector<reco::TrackCollection>::const_iterator it = tracks.begin();
       it != tracks.end(); it++) {    
    TrackDetMatchInfo info = trackAssociator_.associate(theEvent, theEventSetup,trackAssociator_.getFreeTrajectoryState(theEventSetup, (*(*it))), assotiator_parameters);

    math::XYZVector track3Vec(info.trkGlobPosAtEcal.x(),
                              info.trkGlobPosAtEcal.y(),
                              info.trkGlobPosAtEcal.z());

    float deltaR2 = ROOT::Math::VectorUtil::DeltaR(cluster_3vec, track3Vec);
    if (deltaR2 < min_dR) min_dR = deltaR2;          
  }
  return min_dR;
}
