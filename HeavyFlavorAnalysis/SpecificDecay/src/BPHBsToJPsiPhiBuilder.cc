/*
 *  See header file for a description of this class.
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHBsToJPsiPhiBuilder.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHMassSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHChi2Select.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHMassFitSelect.h"
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
BPHBsToJPsiPhiBuilder::BPHBsToJPsiPhiBuilder( const edm::EventSetup& es,
    const std::vector<BPHPlusMinusConstCandPtr>& jpsiCollection,
    const std::vector<BPHPlusMinusConstCandPtr>&  phiCollection ):
  jPsiName( "JPsi" ),
   phiName(  "Phi" ),
  evSetup( &es ),
  jCollection( &jpsiCollection ),
  pCollection( & phiCollection ) {
  jpsiSel = new BPHMassSelect   ( 2.80, 3.40 );
  mphiSel = new BPHMassSelect   ( 1.005, 1.035 );
  massSel = new BPHMassSelect   ( 3.50, 8.00 );
  chi2Sel = new BPHChi2Select   ( 0.02 );
  mFitSel = new BPHMassFitSelect( jPsiName,
                                  BPHParticleMasses::jPsiMass,
                                  BPHParticleMasses::jPsiMWidth,
                                  5.00, 6.00 );
  massConstr = true;
  minPDiff = 1.0e-4;
  updated = false;
}

//--------------
// Destructor --
//--------------
BPHBsToJPsiPhiBuilder::~BPHBsToJPsiPhiBuilder() {
  delete jpsiSel;
  delete mphiSel;
  delete massSel;
  delete chi2Sel;
  delete mFitSel;
}

//--------------
// Operations --
//--------------
vector<BPHRecoConstCandPtr> BPHBsToJPsiPhiBuilder::build() {

  if ( updated ) return bsList;

  BPHRecoBuilder bBs( *evSetup );
  bBs.setMinPDiffererence( minPDiff );
  bBs.add( jPsiName, *jCollection );
  bBs.add(  phiName, *pCollection );
  bBs.filter( jPsiName, *jpsiSel );
  bBs.filter(  phiName, *mphiSel );

  bBs.filter( *massSel );
  bBs.filter( *chi2Sel );
  if ( massConstr ) bBs.filter( *mFitSel );

  bsList = BPHRecoCandidate::build( bBs );
//
//  Apply kinematic constraint on the JPsi mass.
//  The operation is already performed when apply the mass selection,
//  so it's not repeated. The following code is left as example
//  for similar operations
//
//  int iBs;
//  int nBs = ( massConstr ? bsList.size() : 0 );
//  for ( iBs = 0; iBs < nBs; ++iBs ) {
//    BPHRecoCandidate* cptr( const_cast<BPHRecoCandidate*>(
//                            bsList[iBs].get() ) );
//    BPHRecoConstCandPtr jpsi = cptr->getComp( jPsiName );
//    double jMass = jpsi->constrMass();
//    if ( jMass < 0 ) continue;
//    double sigma = jpsi->constrSigma();
//    cptr->kinematicTree( jPsiName, jMass, sigma );
//  }
  updated = true;

  return bsList;

}

/// set cuts
void BPHBsToJPsiPhiBuilder::setJPsiMassMin( double m ) {
  updated = false;
  jpsiSel->setMassMin( m );
  return;
}


void BPHBsToJPsiPhiBuilder::setJPsiMassMax( double m ) {
  updated = false;
  jpsiSel->setMassMax( m );
  return;
}


void BPHBsToJPsiPhiBuilder::setPhiMassMin( double m ) {
  updated = false;
  mphiSel->setMassMin( m );
  return;
}


void BPHBsToJPsiPhiBuilder::setPhiMassMax( double m ) {
  updated = false;
  mphiSel->setMassMax( m );
  return;
}


void BPHBsToJPsiPhiBuilder::setMassMin( double m ) {
  updated = false;
  massSel->setMassMin( m );
  return;
}


void BPHBsToJPsiPhiBuilder::setMassMax( double m ) {
  updated = false;
  massSel->setMassMax( m );
  return;
}


void BPHBsToJPsiPhiBuilder::setProbMin( double p ) {
  updated = false;
  chi2Sel->setProbMin( p );
  return;
}


void BPHBsToJPsiPhiBuilder::setMassFitMin( double m ) {
  updated = false;
  mFitSel->setMassMin( m );
  return;
}


void BPHBsToJPsiPhiBuilder::setMassFitMax( double m ) {
  updated = false;
  mFitSel->setMassMax( m );
  return;
}


void BPHBsToJPsiPhiBuilder::setConstr( bool flag ) {
  updated = false;
  massConstr = flag;
  return;
}

/// get current cuts
double BPHBsToJPsiPhiBuilder::getJPsiMassMin() const {
  return jpsiSel->getMassMin();
}


double BPHBsToJPsiPhiBuilder::getJPsiMassMax() const {
  return jpsiSel->getMassMax();
}


double BPHBsToJPsiPhiBuilder::getPhiMassMin() const {
  return mphiSel->getMassMin();
}


double BPHBsToJPsiPhiBuilder::getPhiMassMax() const {
  return mphiSel->getMassMax();
}


double BPHBsToJPsiPhiBuilder::getMassMin() const {
  return massSel->getMassMin();
}


double BPHBsToJPsiPhiBuilder::getMassMax() const {
  return massSel->getMassMax();
}


double BPHBsToJPsiPhiBuilder::getProbMin() const {
  return chi2Sel->getProbMin();
}


double BPHBsToJPsiPhiBuilder::getMassFitMin() const {
  return mFitSel->getMassMin();
}


double BPHBsToJPsiPhiBuilder::getMassFitMax() const {
  return mFitSel->getMassMax();
}


bool BPHBsToJPsiPhiBuilder::getConstr() const {
  return massConstr;
}

