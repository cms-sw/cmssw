#include "GEMCode/GEMValidation/src/GMTCand.h"

GMTCand::GMTCand()
{}

GMTCand::GMTCand(const GMTCand& rhs)
{}

GMTCand::~GMTCand()
{}


void 
GMTCand::init(const L1MuGMTExtendedCand *t,
	      edm::ESHandle< L1MuTriggerScales > &muScales,
	      edm::ESHandle< L1MuTriggerPtScale > &muPtScale)
{
//   l1gmt = t;
  
//   // keep x and y components non-zero and protect against roundoff.
//   pt = muPtScale->getPtScale()->getLowEdge( t->ptIndex() ) + 1.e-6 ;
//   eta = muScales->getGMTEtaScale()->getCenter( t->etaIndex() ) ;
//   //std::cout<<"gmtetalo="<<muScales->getGMTEtaScale()->getLowEdge(t->etaIndex() )<<std::endl;
//   //std::cout<<"gmtetac="<<eta<<std::endl;
//   phi = normalizedPhi( muScales->getPhiScale()->getLowEdge( t->phiIndex() ) ) ;
//   math::PtEtaPhiMLorentzVector p4( pt, eta, phi, MUON_MASS );
//   pt = p4.pt();
//   q = t->quality();
//   rank = t->rank();
}
