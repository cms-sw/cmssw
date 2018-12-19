#include "CalibTracker/SiStripCommon/interface/ShallowTools.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"


namespace shallow {

CLUSTERMAP 
make_cluster_map( const edm::Event& iEvent, const edm::EDGetTokenT< edmNew::DetSetVector<SiStripCluster> > & cluster_token) {
  CLUSTERMAP clustermap;
  edm::Handle<edmNew::DetSetVector<SiStripCluster> > clusters;
  iEvent.getByToken(cluster_token, clusters);
  
  unsigned int clusterindex = 0;  
  for(auto const& ds : *clusters)
    for(auto const& cluster : ds)
    clustermap.insert( std::make_pair( std::make_pair(ds.detId(),cluster.firstStrip()),
				       clusterindex++));
  return clustermap;
}

int 
findTrackIndex(const edm::Handle<edm::View<reco::Track> >& tracks, const reco::Track* track) {
  edm::View<reco::Track>::const_iterator
    it = tracks->begin(),
    end = tracks->end();
  //Compare addresses
  for(; it!=end; it++) {    if (&(*it)==track) { return it - tracks->begin(); }  }
  return -2;
}

LocalVector 
drift( const StripGeomDetUnit* stripDet, const MagneticField& magfield, const SiStripLorentzAngle& lorentzAngle ) {
  LocalVector lbfield=( stripDet->surface()).toLocal( magfield.inTesla(stripDet->surface().position()));
  float tanLorentzAnglePerTesla = lorentzAngle.getLorentzAngle(stripDet->geographicalId());  
  float driftz = stripDet->specificSurface().bounds().thickness();
  float driftx =-tanLorentzAnglePerTesla * lbfield.y() * driftz;
  float drifty = tanLorentzAnglePerTesla * lbfield.x() * driftz;
  return LocalVector(driftx,drifty,driftz);
}

}
