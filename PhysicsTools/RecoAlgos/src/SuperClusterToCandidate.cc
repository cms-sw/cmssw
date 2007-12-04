#include "PhysicsTools/RecoAlgos/src/SuperClusterToCandidate.h"
using namespace reco;
using namespace converter;

void SuperClusterToCandidate::convert( size_t idx, const edm::Handle<reco::SuperClusterCollection> & superClusters, 
				       RecoEcalCandidate & c ) const {
  SuperClusterRef scRef( superClusters, idx );
  const SuperCluster & sc = * scRef;
  math::XYZPoint v( 0, 0, 0 ); // this should be taken from something else...
  math::XYZVector p = sc.energy() * ( sc.position() - v ).unit();
  double t = sqrt( massSqr_ + p.mag2() );
  c.setCharge( 0 );
  c.setVertex( v );
  c.setP4( Candidate::LorentzVector( p.x(), p.y(), p.z(), t ) );
  c.setSuperCluster( scRef );
  c.setPdgId( particle_.pdgId() );
}
