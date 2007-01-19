// $Id: GenParticleCandidate.cc,v 1.6 2007/01/15 12:33:59 llista Exp $
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include <CLHEP/HepMC/GenParticle.h>
#include <CLHEP/HepMC/GenVertex.h>
#include <iostream>
using namespace reco;

GenParticleCandidate::GenParticleCandidate( const HepMC::GenParticle * p ) : 
  CompositeRefCandidate(), 
  pdgId_( p->pdg_id() ), 
  status_( p->status() ) {
  q_ = p->particleID().threeCharge() / 3;
  CLHEP::HepLorentzVector p4 =p->momentum();
  p4_ = LorentzVector( p4.x(), p4.y(), p4.z(), p4.t() );
  const HepMC::GenVertex * v = p->production_vertex();
  if ( v != 0 ) {
    HepGeom::Point3D<double> vtx = v->point3d();
    vertex_ = Point( vtx.x() / 10. , vtx.y() / 10. , vtx.z() / 10. );
  } else {
    vertex_.SetXYZ( 0, 0, 0 );
  }
  // don't fill references to daughters at this level
}

GenParticleCandidate::GenParticleCandidate( Charge q, const LorentzVector & p4, 
					    const Point & vtx, int pdgId, int status ) : 
  CompositeRefCandidate( q, p4, vtx ), 
  pdgId_( pdgId ), 
  status_( status ) {
}

GenParticleCandidate::~GenParticleCandidate() { }

bool GenParticleCandidate::overlap( const Candidate & c ) const {
  return & c == this;
}

GenParticleCandidate * GenParticleCandidate::clone() const {
  return new GenParticleCandidate( * this );
}
