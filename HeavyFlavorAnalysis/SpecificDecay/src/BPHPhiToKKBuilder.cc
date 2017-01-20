/*
 *  See header file for a description of this class.
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHPhiToKKBuilder.h"

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
using namespace std;

//-------------------
// Initializations --
//-------------------


//----------------
// Constructors --
//----------------
BPHPhiToKKBuilder::BPHPhiToKKBuilder(
               const edm::EventSetup& es,
               const BPHRecoBuilder::BPHGenericCollection* kPosCollection,
               const BPHRecoBuilder::BPHGenericCollection* kNegCollection ):
  kPosName( "KPos" ),
  kNegName( "KNeg" ),
  evSetup( &es ),
  posCollection( kPosCollection ),
  negCollection( kNegCollection ) {
    ptSel = new BPHParticlePtSelect (  0.7 );
   etaSel = new BPHParticleEtaSelect( 10.0 );
  massSel = new BPHMassSelect( 1.0, 1.04 );
  chi2Sel = new BPHChi2Select( 0.0 );
  updated = false;
}

//--------------
// Destructor --
//--------------
BPHPhiToKKBuilder::~BPHPhiToKKBuilder() {
  delete   ptSel;
  delete  etaSel;
  delete massSel;
  delete chi2Sel;
}

//--------------
// Operations --
//--------------
vector<BPHPlusMinusConstCandPtr> BPHPhiToKKBuilder::build() {

  if ( updated ) return phiList;

  BPHRecoBuilder bPhi( *evSetup );
  bPhi.add( kPosName, posCollection, BPHParticleMasses::kaonMass,
                                     BPHParticleMasses::kaonMSigma );
  bPhi.add( kNegName, negCollection, BPHParticleMasses::kaonMass,
                                     BPHParticleMasses::kaonMSigma );
  bPhi.filter( kPosName, *ptSel );
  bPhi.filter( kNegName, *ptSel );
  bPhi.filter( kPosName, *etaSel );
  bPhi.filter( kNegName, *etaSel );

  bPhi.filter( *massSel );
  bPhi.filter( *chi2Sel );

  phiList = BPHPlusMinusCandidate::build( bPhi, kPosName, kNegName );
  updated = true;
  return phiList;

}

/// set cuts
void BPHPhiToKKBuilder::setPtMin( double pt ) {
  updated = false;
  ptSel->setPtMin( pt );
  return;
}


void BPHPhiToKKBuilder::setEtaMax( double eta ) {
  updated = false;
  etaSel->setEtaMax( eta );
  return;
}


void BPHPhiToKKBuilder::setMassMin( double m ) {
  updated = false;
  massSel->setMassMin( m );
  return;
}


void BPHPhiToKKBuilder::setMassMax( double m ) {
  updated = false;
  massSel->setMassMax( m );
  return;
}


void BPHPhiToKKBuilder::setProbMin( double p ) {
  updated = false;
  chi2Sel->setProbMin( p );
  return;
}


void BPHPhiToKKBuilder::setConstr( double mass, double sigma ) {
  updated = false;
  cMass  = mass;
  cSigma = sigma;
  return;
}

/// get current cuts
double BPHPhiToKKBuilder::getPtMin() const {
  return ptSel->getPtMin();
}


double BPHPhiToKKBuilder::getEtaMax() const {
  return etaSel->getEtaMax();
}


double BPHPhiToKKBuilder::getMassMin() const {
  return massSel->getMassMin();
}


double BPHPhiToKKBuilder::getMassMax() const {
  return massSel->getMassMax();
}


double BPHPhiToKKBuilder::getProbMin() const {
  return chi2Sel->getProbMin();
}


double BPHPhiToKKBuilder::getConstrMass() const {
  return cMass;
}


double BPHPhiToKKBuilder::getConstrSigma() const {
  return cSigma;
}

