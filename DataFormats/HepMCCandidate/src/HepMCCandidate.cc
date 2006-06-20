// $Id: HepMCCandidate.cc,v 1.4 2006/04/07 11:38:37 llista Exp $
#include "DataFormats/HepMCCandidate/interface/HepMCCandidate.h"
#include <CLHEP/HepMC/GenParticle.h>
#include <CLHEP/HepMC/GenVertex.h>
#include <iostream>
using namespace reco;

HepMCCandidate::HepMCCandidate( const HepMC::GenParticle * p ) : 
  LeafCandidate(), genParticle_( p ) {
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
}

HepMCCandidate::~HepMCCandidate() { }

bool HepMCCandidate::overlap( const Candidate & c ) const {
  const HepMCCandidate * mcc = dynamic_cast<const HepMCCandidate *>( & c );
  if ( mcc == 0 ) return false;
  const HepMC::GenParticle * g1 = genParticle(), * g2 = mcc->genParticle();
  return g1 != 0 && g2 != 0 && g1 == g2;
  /**
   * WARNING: should also check here the full decay chain 
   * for possible overlap of daughters
   */
}

HepMCCandidate * HepMCCandidate::clone() const {
  return new HepMCCandidate( * this );
}
