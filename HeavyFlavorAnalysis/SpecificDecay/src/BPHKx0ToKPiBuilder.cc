/*
 *  See header file for a description of this class.
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHKx0ToKPiBuilder.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticlePtSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticleEtaSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHMassSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHChi2Select.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticleMasses.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>
using namespace std;

//-------------------
// Initializations --
//-------------------


//----------------
// Constructors --
//----------------
BPHKx0ToKPiBuilder::BPHKx0ToKPiBuilder(
               const edm::EventSetup& es,
               const BPHRecoBuilder::BPHGenericCollection* kaonCollection,
               const BPHRecoBuilder::BPHGenericCollection* pionCollection ):
  kaonName( "Kaon" ),
  pionName( "Pion" ),
  evSetup( &es ),
  kCollection( kaonCollection ),
  pCollection( pionCollection ) {
    ptSel = new BPHParticlePtSelect (  0.7 );
   etaSel = new BPHParticleEtaSelect( 10.0 );
  massSel = new BPHMassSelect( 0.75, 1.05 );
  chi2Sel = new BPHChi2Select( 0.0 );
  updated = false;
}

//--------------
// Destructor --
//--------------
BPHKx0ToKPiBuilder::~BPHKx0ToKPiBuilder() {
  delete   ptSel;
  delete  etaSel;
  delete massSel;
  delete chi2Sel;
}

//--------------
// Operations --
//--------------
vector<BPHPlusMinusConstCandPtr> BPHKx0ToKPiBuilder::build() {

  if ( updated ) return kx0List;

  BPHRecoBuilder bKx0( *evSetup );
  bKx0.add( kaonName, kCollection, BPHParticleMasses::kaonMass,
                                   BPHParticleMasses::kaonMSigma );
  bKx0.add( pionName, pCollection, BPHParticleMasses::pionMass,
                                   BPHParticleMasses::pionMSigma );
  bKx0.filter( kaonName, *ptSel );
  bKx0.filter( pionName, *ptSel );
  bKx0.filter( kaonName, *etaSel );
  bKx0.filter( pionName, *etaSel );

  bKx0.filter( *chi2Sel );

  vector<BPHPlusMinusConstCandPtr>
  tmpList = BPHPlusMinusCandidate::build( bKx0, kaonName, pionName );

  int ikx;
  int nkx = tmpList.size();
  kx0List.clear();
  kx0List.reserve( nkx );
  for ( ikx = 0; ikx < nkx; ++ikx ) {
    BPHPlusMinusConstCandPtr& px0 = tmpList[ikx];
    const
    BPHPlusMinusCandidate* kx0 = px0.get();
    BPHPlusMinusCandidate* kxb = new BPHPlusMinusCandidate( evSetup );
    kxb->add( pionName, kx0->originalReco( kx0->getDaug( kaonName ) ),
              BPHParticleMasses::pionMass );
    kxb->add( kaonName, kx0->originalReco( kx0->getDaug( pionName ) ),
              BPHParticleMasses::kaonMass );
    BPHPlusMinusConstCandPtr pxb( kxb );
    if ( fabs( kx0->composite().mass() - BPHParticleMasses::kx0Mass ) <
         fabs( kxb->composite().mass() - BPHParticleMasses::kx0Mass ) ) {
      if ( !massSel->accept( *kx0 ) ) continue;
      kx0List.push_back( px0 );
    }
    else {
      if ( !massSel->accept( *kxb ) ) continue;
      kx0List.push_back( pxb );
    }
  }

  updated = true;
  return kx0List;

}

/// set cuts
void BPHKx0ToKPiBuilder::setPtMin( double pt ) {
  updated = false;
  ptSel->setPtMin( pt );
  return;
}


void BPHKx0ToKPiBuilder::setEtaMax( double eta ) {
  updated = false;
  etaSel->setEtaMax( eta );
  return;
}


void BPHKx0ToKPiBuilder::setMassMin( double m ) {
  updated = false;
  massSel->setMassMin( m );
  return;
}


void BPHKx0ToKPiBuilder::setMassMax( double m ) {
  updated = false;
  massSel->setMassMax( m );
  return;
}


void BPHKx0ToKPiBuilder::setProbMin( double p ) {
  updated = false;
  chi2Sel->setProbMin( p );
  return;
}


void BPHKx0ToKPiBuilder::setConstr( double mass, double sigma ) {
  updated = false;
  cMass  = mass;
  cSigma = sigma;
  return;
}

/// get current cuts
double BPHKx0ToKPiBuilder::getPtMin() const {
  return ptSel->getPtMin();
}


double BPHKx0ToKPiBuilder::getEtaMax() const {
  return etaSel->getEtaMax();
}


double BPHKx0ToKPiBuilder::getMassMin() const {
  return massSel->getMassMin();
}


double BPHKx0ToKPiBuilder::getMassMax() const {
  return massSel->getMassMax();
}


double BPHKx0ToKPiBuilder::getProbMin() const {
  return chi2Sel->getProbMin();
}


double BPHKx0ToKPiBuilder::getConstrMass() const {
  return cMass;
}


double BPHKx0ToKPiBuilder::getConstrSigma() const {
  return cSigma;
}

