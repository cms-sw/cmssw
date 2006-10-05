// $Id: HepMCCandidate.cc,v 1.5 2006/06/20 14:53:18 llista Exp $
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
    vertex_ = Point( vtx.x(), vtx.y(), vtx.z() );
  } else {
    vertex_.SetXYZ( 0, 0, 0 );
  }
  // don't fill references to daughters at this level
}

GenParticleCandidate::~GenParticleCandidate() { }

bool GenParticleCandidate::overlap( const Candidate & c ) const {
  return & c == this;
}

GenParticleCandidate * GenParticleCandidate::clone() const {
  return new GenParticleCandidate( * this );
}
