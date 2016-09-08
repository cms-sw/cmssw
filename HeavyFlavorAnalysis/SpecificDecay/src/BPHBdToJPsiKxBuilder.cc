/*
 *  See header file for a description of this class.
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHBdToJPsiKxBuilder.h"

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
BPHBdToJPsiKxBuilder::BPHBdToJPsiKxBuilder( const edm::EventSetup& es,
    const std::vector<BPHPlusMinusConstCandPtr>& jpsiCollection,
    const std::vector<BPHPlusMinusConstCandPtr>&  kx0Collection ):
  jPsiName( "JPsi" ),
   kx0Name(  "Kx0" ),
  evSetup( &es ),
  jCollection( &jpsiCollection ),
  kCollection( & kx0Collection ) {
  jpsiSel = new BPHMassSelect   ( 2.80, 3.40 );
  mkx0Sel = new BPHMassSelect   ( 0.80, 1.00 );
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
BPHBdToJPsiKxBuilder::~BPHBdToJPsiKxBuilder() {
  delete jpsiSel;
  delete mkx0Sel;
  delete massSel;
  delete chi2Sel;
  delete mFitSel;
}

//--------------
// Operations --
//--------------
vector<BPHRecoConstCandPtr> BPHBdToJPsiKxBuilder::build() {

  if ( updated ) return bdList;

  BPHRecoBuilder bBd( *evSetup );
  bBd.setMinPDiffererence( minPDiff );
  bBd.add( jPsiName, *jCollection );
  bBd.add(  kx0Name, *kCollection );
  bBd.filter( jPsiName, *jpsiSel );
  bBd.filter(  kx0Name, *mkx0Sel );

  bBd.filter( *massSel );
  bBd.filter( *chi2Sel );
  if ( massConstr ) bBd.filter( *mFitSel );

  bdList = BPHRecoCandidate::build( bBd );
//
//  Apply kinematic constraint on the JPsi mass.
//  The operation is already performed when apply the mass selection,
//  so it's not repeated. The following code is left as example
//  for similar operations
//
//  int iBd;
//  int nBd = ( massConstr ? bdList.size() : 0 );
//  for ( iBd = 0; iBd < nBd; ++iBd ) {
//    BPHRecoCandidate* cptr( const_cast<BPHRecoCandidate*>(
//                            bdList[iBd].get() ) );
//    BPHRecoConstCandPtr jpsi = cptr->getComp( jPsiName );
//    double jMass = jpsi->constrMass();
//    if ( jMass < 0 ) continue;
//    double sigma = jpsi->constrSigma();
//    cptr->kinematicTree( jPsiName, jMass, sigma );
//  }
  updated = true;

  return bdList;

}

/// set cuts
void BPHBdToJPsiKxBuilder::setJPsiMassMin( double m ) {
  updated = false;
  jpsiSel->setMassMin( m );
  return;
}


void BPHBdToJPsiKxBuilder::setJPsiMassMax( double m ) {
  updated = false;
  jpsiSel->setMassMax( m );
  return;
}


void BPHBdToJPsiKxBuilder::setKxMassMin( double m ) {
  updated = false;
  mkx0Sel->setMassMin( m );
  return;
}


void BPHBdToJPsiKxBuilder::setKxMassMax( double m ) {
  updated = false;
  mkx0Sel->setMassMax( m );
  return;
}


void BPHBdToJPsiKxBuilder::setMassMin( double m ) {
  updated = false;
  massSel->setMassMin( m );
  return;
}


void BPHBdToJPsiKxBuilder::setMassMax( double m ) {
  updated = false;
  massSel->setMassMax( m );
  return;
}


void BPHBdToJPsiKxBuilder::setProbMin( double p ) {
  updated = false;
  chi2Sel->setProbMin( p );
  return;
}


void BPHBdToJPsiKxBuilder::setMassFitMin( double m ) {
  updated = false;
  mFitSel->setMassMin( m );
  return;
}


void BPHBdToJPsiKxBuilder::setMassFitMax( double m ) {
  updated = false;
  mFitSel->setMassMax( m );
  return;
}


void BPHBdToJPsiKxBuilder::setConstr( bool flag ) {
  updated = false;
  massConstr = flag;
  return;
}

/// get current cuts
double BPHBdToJPsiKxBuilder::getJPsiMassMin() const {
  return jpsiSel->getMassMin();
}


double BPHBdToJPsiKxBuilder::getJPsiMassMax() const {
  return jpsiSel->getMassMax();
}


double BPHBdToJPsiKxBuilder::getKxMassMin() const {
  return mkx0Sel->getMassMin();
}


double BPHBdToJPsiKxBuilder::getKxMassMax() const {
  return mkx0Sel->getMassMax();
}


double BPHBdToJPsiKxBuilder::getMassMin() const {
  return massSel->getMassMin();
}


double BPHBdToJPsiKxBuilder::getMassMax() const {
  return massSel->getMassMax();
}


double BPHBdToJPsiKxBuilder::getProbMin() const {
  return chi2Sel->getProbMin();
}


double BPHBdToJPsiKxBuilder::getMassFitMin() const {
  return mFitSel->getMassMin();
}


double BPHBdToJPsiKxBuilder::getMassFitMax() const {
  return mFitSel->getMassMax();
}


bool BPHBdToJPsiKxBuilder::getConstr() const {
  return massConstr;
}

