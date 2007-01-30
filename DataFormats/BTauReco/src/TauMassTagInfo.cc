#include "DataFormats/BTauReco/interface/TauMassTagInfo.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Common/interface/RefVector.h"

using namespace edm;
using namespace reco;
using namespace std;

double reco::TauMassTagInfo::discriminator(const double rm_cone,const double pt_cut,
                                   const double rs_cone, const double track_cone, 
                                   const double m_cut) const {
  double discriminator = 0.0;
  double invariantMass = getInvariantMass(rm_cone,pt_cut,rs_cone,track_cone);  
  if (invariantMass >= 0.0 && invariantMass < m_cut ) discriminator = 1.0;
  return discriminator;
}
//
// --default discriminator: returns the value of the discriminator of the jet tag
//
double reco::TauMassTagInfo::discriminator() const {

  return isolatedTau->jetRef()->discriminator(); 

}
//
// -- Set IsolatedTauTag
//
void reco::TauMassTagInfo::setIsolatedTauTag(const IsolatedTauTagInfoRef isolationRef){
   isolatedTau = isolationRef;
}
//
// -- Get IsolatedTauTag
//
const IsolatedTauTagInfoRef& reco::TauMassTagInfo::getIsolatedTauTag() const {
   return isolatedTau;
}
//
// -- Set JetTag
//
void reco::TauMassTagInfo::setJetTag(const JetTagRef jetTagRef){
  jetTag = jetTagRef;
}
//
// -- get JetTag
//
const JetTagRef & reco::TauMassTagInfo::getJetTag() const {
  return jetTag;
}
//
// -- Set Cluster Collection
//
void reco::TauMassTagInfo::storeClusterTrackCollection(reco::BasicClusterRef clusterRef,float dr) {
  
  clusterMap.insert(clusterRef, dr);
}

//void reco::TauMassTagInfo::setBasicClusterCollection(const BasicClusterCollection& clusters){
//  basicClusters = clusters;
//}
//
// -- Set Basic Cluster Collection
//
//const BasicClusterCollection& reco::TauMassTagInfo::getBasicClusterCollection() const {
//  return basicClusters;
//}
//
// -- Get Invariant Mass
//
double reco::TauMassTagInfo::getInvariantMass(const double rm_cone, const double pt_cut, 
                        const double rs_cone, const double track_cone) const {
  double invariantMass = -1.0;

  const TrackRef leadTk= isolatedTau->leadingSignalTrack(rm_cone, pt_cut);
  if(!leadTk) {
    cout <<" TauMassTagInfo::  No Leading Track !!  " << endl;    
    return invariantMass;
  }
  math::XYZVector momentum = (*leadTk).momentum();
  const RefVector<TrackCollection> signalTracks = isolatedTau->tracksInCone(momentum,rs_cone,pt_cut);
  if (signalTracks.size() == 0) return invariantMass;

  double px_inv = 0.0;
  double py_inv = 0.0;
  double pz_inv = 0.0;
  double e_inv  = 0.0;
  for(RefVector<TrackCollection>::const_iterator itrack = signalTracks.begin(); 
               itrack != signalTracks.end(); itrack++) {
    px_inv += (*itrack)->px();
    py_inv += (*itrack)->py();
    pz_inv += (*itrack)->pz();
    e_inv += (*itrack)->p();
  }
  
  // Add Clusters away from tracks

  for(ClusterTrackAssociationCollection::const_iterator mapIter = clusterMap.begin(); mapIter != clusterMap.end(); mapIter++) {
    const reco::BasicClusterRef & iclus = mapIter->key;
    float dr  = mapIter->val;
    if (dr > track_cone) {
      math::RhoEtaPhiVectorD clusREP(iclus->energy(), iclus->eta(), iclus->phi());
      px_inv += clusREP.X();
      py_inv += clusREP.Y();
      pz_inv += clusREP.Z();
      e_inv  += iclus->energy();
    }           
  }

  invariantMass = e_inv*e_inv - px_inv*px_inv - py_inv*py_inv - pz_inv*pz_inv;
  if (invariantMass >0.0) invariantMass = sqrt(invariantMass);
  else {
    cout << "TauMassTagInfo::(E,Px,Py,Pz): " << e_inv << " " << px_inv << " " << py_inv << " " << pz_inv << endl;
    invariantMass = -1.0;
  }
  return invariantMass;
}
