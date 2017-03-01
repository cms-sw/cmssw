/*
 *  See header file for a description of this class.
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHBuToJPsiKBuilder.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticleNeutralVeto.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticlePtSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticleEtaSelect.h"
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
BPHBuToJPsiKBuilder::BPHBuToJPsiKBuilder( const edm::EventSetup& es,
    const std::vector<BPHPlusMinusConstCandPtr>& jpsiCollection,
    const BPHRecoBuilder::BPHGenericCollection*  kaonCollection ):
  jPsiName( "JPsi" ),
  kaonName( "Kaon" ),
  evSetup( &es ),
  jCollection( &jpsiCollection ),
  kCollection(  kaonCollection )  {
  jpsiSel = new BPHMassSelect       ( 2.80, 3.40 );
   knVeto = new BPHParticleNeutralVeto;
    ptSel = new BPHParticlePtSelect (  0.7 );
   etaSel = new BPHParticleEtaSelect( 10.0 );
  massSel = new BPHMassSelect       ( 3.50, 8.00 );
  chi2Sel = new BPHChi2Select       ( 0.02 );
  mFitSel = new BPHMassFitSelect    ( jPsiName,
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
BPHBuToJPsiKBuilder::~BPHBuToJPsiKBuilder() {
  delete jpsiSel;
  delete  knVeto;
  delete   ptSel;
  delete  etaSel;
  delete massSel;
  delete chi2Sel;
  delete mFitSel;
}

//--------------
// Operations --
//--------------
vector<BPHRecoConstCandPtr> BPHBuToJPsiKBuilder::build() {

  if ( updated ) return buList;

  BPHRecoBuilder bBu( *evSetup );
  bBu.setMinPDiffererence( minPDiff );
  bBu.add( jPsiName, *jCollection );
  bBu.add( kaonName,  kCollection, BPHParticleMasses::kaonMass,
                                   BPHParticleMasses::kaonMSigma );
  bBu.filter( jPsiName, *jpsiSel );
  bBu.filter( kaonName, * knVeto );
  bBu.filter( kaonName, *  ptSel );
  bBu.filter( kaonName, * etaSel );

  bBu.filter( *massSel );
  bBu.filter( *chi2Sel );
  if ( massConstr ) bBu.filter( *mFitSel );

  buList = BPHRecoCandidate::build( bBu );
//
//  Apply kinematic constraint on the JPsi mass.
//  The operation is already performed when apply the mass selection,
//  so it's not repeated. The following code is left as example
//  for similar operations
//
//  int iBu;
//  int nBu = ( massConstr ? buList.size() : 0 );
//  for ( iBu = 0; iBu < nBu; ++iBu ) {
//    BPHRecoCandidate* cptr( const_cast<BPHRecoCandidate*>(
//                            buList[iBu].get() ) );
//    BPHRecoConstCandPtr jpsi = cptr->getComp( jPsiName );
//    double jMass = jpsi->constrMass();
//    if ( jMass < 0 ) continue;
//    double sigma = jpsi->constrSigma();
//    cptr->kinematicTree( jPsiName, jMass, sigma );
//  }
  updated = true;
  return buList;

}

/// set cuts
void BPHBuToJPsiKBuilder::setJPsiMassMin( double m ) {
  updated = false;
  jpsiSel->setMassMin( m );
  return;
}


void BPHBuToJPsiKBuilder::setJPsiMassMax( double m ) {
  updated = false;
  jpsiSel->setMassMax( m );
  return;
}


void BPHBuToJPsiKBuilder::setKPtMin( double pt ) {
  updated = false;
  ptSel->setPtMin( pt );
  return;
}


void BPHBuToJPsiKBuilder::setKEtaMax( double eta ) {
  updated = false;
  etaSel->setEtaMax( eta );
  return;
}


void BPHBuToJPsiKBuilder::setMassMin( double m ) {
  updated = false;
  massSel->setMassMin( m );
  return;
}


void BPHBuToJPsiKBuilder::setMassMax( double m ) {
  updated = false;
  massSel->setMassMax( m );
  return;
}


void BPHBuToJPsiKBuilder::setProbMin( double p ) {
  updated = false;
  chi2Sel->setProbMin( p );
  return;
}


void BPHBuToJPsiKBuilder::setMassFitMin( double m ) {
  updated = false;
  mFitSel->setMassMin( m );
  return;
}


void BPHBuToJPsiKBuilder::setMassFitMax( double m ) {
  updated = false;
  mFitSel->setMassMax( m );
  return;
}


void BPHBuToJPsiKBuilder::setConstr( bool flag ) {
  updated = false;
  massConstr = flag;
  return;
}

/// get current cuts
double BPHBuToJPsiKBuilder::getJPsiMassMin() const {
  return jpsiSel->getMassMin();
}


double BPHBuToJPsiKBuilder::getJPsiMassMax() const {
  return jpsiSel->getMassMax();
}


double BPHBuToJPsiKBuilder::getKPtMin() const {
  return ptSel->getPtMin();
}


double BPHBuToJPsiKBuilder::getKEtaMax() const {
  return etaSel->getEtaMax();
}


double BPHBuToJPsiKBuilder::getMassMin() const {
  return massSel->getMassMin();
}


double BPHBuToJPsiKBuilder::getMassMax() const {
  return massSel->getMassMax();
}


double BPHBuToJPsiKBuilder::getProbMin() const {
  return chi2Sel->getProbMin();
}


double BPHBuToJPsiKBuilder::getMassFitMin() const {
  return mFitSel->getMassMin();
}


double BPHBuToJPsiKBuilder::getMassFitMax() const {
  return mFitSel->getMassMax();
}


bool BPHBuToJPsiKBuilder::getConstr() const {
  return massConstr;
}

