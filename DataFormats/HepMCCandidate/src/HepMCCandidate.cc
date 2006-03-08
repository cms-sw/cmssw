// $Id: RecoCandidate.cc,v 1.1 2006/02/28 10:59:15 llista Exp $
#include "DataFormats/HepMCCandidate/interface/HepMCCandidate.h"
#include <CLHEP/HepMC/GenParticle.h>
#include <CLHEP/HepMC/GenVertex.h>

using namespace reco;

HepMCCandidate::HepMCCandidate( const HepMC::GenParticle * p ) : 
  LeafCandidate(), genParticle_( p ) {
  q_ = Charge( p->particledata().charge() );
  CLHEP::HepLorentzVector p4 =p->momentum();
  p4_ = LorentzVector( p4.x(), p4.y(), p4.z(), p4.t() );
  HepGeom::Point3D<double> vtx = p->production_vertex()->point3d();
  vtx_ = Point( vtx.x(), vtx.y(), vtx.z() );
}

HepMCCandidate::~HepMCCandidate() { }

bool HepMCCandidate::overlap( const Candidate & c ) const {
  const HepMCCandidate * mcc = dynamic_cast<const HepMCCandidate *>( & c );
  if ( mcc == 0 ) return false;
  const HepMC::GenParticle * g1 = genParticle(), * g2 = mcc->genParticle();
  return g1 != 0 && g2 != 0 && g1 == g2;
}
