/*
 *  See header file for a description of this class.
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHOniaToMuMuBuilder.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoSelect.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHMuonPtSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHMuonEtaSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHMassSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHChi2Select.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHMultiSelect.h"
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
BPHOniaToMuMuBuilder::BPHOniaToMuMuBuilder(
    const edm::EventSetup& es,
    const BPHRecoBuilder::BPHGenericCollection* muPosCollection,
    const BPHRecoBuilder::BPHGenericCollection* muNegCollection ):
  muPosName( "MuPos" ),
  muNegName( "MuNeg" ),
  evSetup( &es ),
  posCollection( muPosCollection ),
  negCollection( muNegCollection ) {
  setParameters( Phi , 3.0, 10.0,  0.50,  1.50, 0.0,
                 BPHParticleMasses:: phiMass, BPHParticleMasses:: phiMWidth );
  setParameters( Psi1, 3.0, 10.0,  2.00,  3.40, 0.0,
                 BPHParticleMasses::jPsiMass, BPHParticleMasses::jPsiMWidth );
  setParameters( Psi2, 3.0, 10.0,  3.40,  6.00, 0.0,
                 BPHParticleMasses::psi2Mass, BPHParticleMasses::psi2MWidth );
  setParameters( Ups , 3.0, 10.0,  6.00, 12.00, 0.0,
                 -1.0     , 0.0      );
  setParameters( Ups1, 3.0, 10.0,  6.00,  9.75, 0.0,
                 BPHParticleMasses::ups1Mass, BPHParticleMasses::ups1MWidth );
  setParameters( Ups2, 3.0, 10.0,  9.75, 10.20, 0.0,
                 BPHParticleMasses::ups2Mass, BPHParticleMasses::ups2MWidth );
  setParameters( Ups3, 3.0, 10.0, 10.20, 12.00, 0.0,
                 BPHParticleMasses::ups3Mass, BPHParticleMasses::ups3MWidth );
  updated = false;
}

//--------------
// Destructor --
//--------------
BPHOniaToMuMuBuilder::~BPHOniaToMuMuBuilder() {
  map< oniaType, OniaParameters >::iterator iter = oniaPar.begin();
  map< oniaType, OniaParameters >::iterator iend = oniaPar.end();
  while ( iter != iend ) {
    OniaParameters& par = iter++->second;
    delete par.  ptSel;
    delete par. etaSel;
    delete par.massSel;
    delete par.chi2Sel;
  }
}

//--------------
// Operations --
//--------------
vector<BPHPlusMinusConstCandPtr> BPHOniaToMuMuBuilder::build() {

  if ( updated ) return fullList;

  BPHMultiSelect<BPHRecoSelect    >  ptSel( BPHSelectOperation::or_mode );
  BPHMultiSelect<BPHRecoSelect    > etaSel( BPHSelectOperation::or_mode );
  BPHMultiSelect<BPHMomentumSelect>   mSel( BPHSelectOperation::or_mode );
  BPHMultiSelect<BPHVertexSelect  >   vSel( BPHSelectOperation::or_mode );

  map< oniaType, OniaParameters >::iterator iter = oniaPar.begin();
  map< oniaType, OniaParameters >::iterator iend = oniaPar.end();
  while ( iter != iend ) {
    OniaParameters& par = iter++->second;
     ptSel.include( *par.  ptSel );
    etaSel.include( *par. etaSel );
      mSel.include( *par.massSel );
      vSel.include( *par.chi2Sel );
  }

  BPHRecoBuilder bOnia( *evSetup );
  bOnia.add( muPosName, posCollection, BPHParticleMasses::muonMass,
                                       BPHParticleMasses::muonMSigma );
  bOnia.add( muNegName, negCollection, BPHParticleMasses::muonMass,
                                       BPHParticleMasses::muonMSigma );
  bOnia.filter( muPosName,  ptSel );
  bOnia.filter( muNegName,  ptSel );
  bOnia.filter( muPosName, etaSel );
  bOnia.filter( muNegName, etaSel );
  bOnia.filter( mSel );
  bOnia.filter( vSel );

  fullList = BPHPlusMinusCandidate::build( bOnia, muPosName, muNegName );
  updated = true;
  return fullList;

}


vector<BPHPlusMinusConstCandPtr> BPHOniaToMuMuBuilder::getList(
                                 oniaType type,
                                 BPHRecoSelect    * dSel,
                                 BPHMomentumSelect* mSel,
                                 BPHVertexSelect  * vSel,
                                 BPHFitSelect     * kSel ) {
  extractList( type );
  vector<BPHPlusMinusConstCandPtr>& list = oniaList[type];
  int i;
  int n = list.size();
  vector<BPHPlusMinusConstCandPtr> lsub;
  lsub.reserve( n );
  for ( i = 0; i < n; ++i ) {
    BPHPlusMinusConstCandPtr ptr = list[i];
    const reco::Candidate* muPos = ptr->originalReco( ptr->getDaug(
                           muPosName ) );
    const reco::Candidate* muNeg = ptr->originalReco( ptr->getDaug(
                           muNegName ) );
    if ( ( dSel != 0 ) && ( !dSel->accept( *muPos ) ) ) continue;
    if ( ( dSel != 0 ) && ( !dSel->accept( *muNeg ) ) ) continue;
    if ( ( mSel != 0 ) && ( !mSel->accept( *ptr   ) ) ) continue;
    if ( ( vSel != 0 ) && ( !vSel->accept( *ptr   ) ) ) continue;
    if ( ( kSel != 0 ) && ( !kSel->accept( *ptr   ) ) ) continue;
    lsub.push_back( list[i] );
  }
  return lsub;
}


BPHPlusMinusConstCandPtr BPHOniaToMuMuBuilder::getOriginalCandidate( 
                         const BPHRecoCandidate& cand ) {
  const reco::Candidate* mp = cand.originalReco( cand.getDaug( muPosName ) );
  const reco::Candidate* mn = cand.originalReco( cand.getDaug( muNegName ) );
  int nc = fullList.size();
  int ic;
  for ( ic = 0; ic < nc; ++ ic ) {
    BPHPlusMinusConstCandPtr pmp = fullList[ic];
    const BPHPlusMinusCandidate* pmc = pmp.get();
    if ( pmc->originalReco( pmc->getDaug( muPosName ) ) != mp ) continue;
    if ( pmc->originalReco( pmc->getDaug( muNegName ) ) != mn ) continue;
    return pmp;
  }
  return BPHPlusMinusConstCandPtr( 0 );
}

/// set cuts
void BPHOniaToMuMuBuilder::setPtMin( oniaType type, double pt ) {
  setNotUpdated();
  OniaParameters& par = oniaPar[type];
  par.ptSel->setPtMin( pt );
  return;
}


void BPHOniaToMuMuBuilder::setEtaMax( oniaType type, double eta ) {
  setNotUpdated();
  OniaParameters& par = oniaPar[type];
  par.etaSel->setEtaMax( eta );
  return;
}


void BPHOniaToMuMuBuilder::setMassMin( oniaType type, double m ) {
  setNotUpdated();
  OniaParameters& par = oniaPar[type];
  par.massSel->setMassMin( m );
  return;
}


void BPHOniaToMuMuBuilder::setMassMax( oniaType type, double m ) {
  setNotUpdated();
  OniaParameters& par = oniaPar[type];
  par.massSel->setMassMax( m );
  return;
}


void BPHOniaToMuMuBuilder::setProbMin( oniaType type, double p ) {
  setNotUpdated();
  OniaParameters& par = oniaPar[type];
  par.chi2Sel->setProbMin( p );
  return;
}


void BPHOniaToMuMuBuilder::setConstr( oniaType type,
                                      double mass, double sigma ) {
  setNotUpdated();
  OniaParameters& par = oniaPar[type];
  par.mass  = mass;
  par.sigma = sigma;
  return;
}

/// get current cuts
double BPHOniaToMuMuBuilder::getPtMin( oniaType type ) const {
  const OniaParameters& par = oniaPar.at( type );
  return par.ptSel->getPtMin();
}


double BPHOniaToMuMuBuilder::getEtaMax( oniaType type ) const {
  const OniaParameters& par = oniaPar.at( type );
  return par.etaSel->getEtaMax();
}


double BPHOniaToMuMuBuilder::getMassMin( oniaType type ) const {
  const OniaParameters& par = oniaPar.at( type );
  return par.massSel->getMassMin();
}


double BPHOniaToMuMuBuilder::getMassMax( oniaType type ) const {
  const OniaParameters& par = oniaPar.at( type );
  return par.massSel->getMassMax();
}


double BPHOniaToMuMuBuilder::getProbMin( oniaType type ) const {
  const OniaParameters& par = oniaPar.at( type );
  return par.chi2Sel->getProbMin();
}


double BPHOniaToMuMuBuilder::getConstrMass( oniaType type ) const {
  const OniaParameters& par = oniaPar.at( type );
  return par.mass;
}


double BPHOniaToMuMuBuilder::getConstrSigma( oniaType type ) const {
  const OniaParameters& par = oniaPar.at( type );
  return par.sigma;
}


void BPHOniaToMuMuBuilder::setNotUpdated() {
  map< oniaType, OniaParameters >::iterator iter = oniaPar.begin();
  map< oniaType, OniaParameters >::iterator iend = oniaPar.end();
  while ( iter != iend ) iter++->second.updated = false;
  updated = false;
  return;
}


void BPHOniaToMuMuBuilder::setParameters( oniaType type,
                                          double ptMin, double etaMax,
                                          double massMin, double massMax,
                                          double probMin,
                                          double mass, double sigma ) {
  OniaParameters& par = oniaPar[type];
  par.  ptSel = new BPHMuonPtSelect( ptMin );
  par. etaSel = new BPHMuonEtaSelect( etaMax );
  par.massSel = new BPHMassSelect( massMin, massMax );
  par.chi2Sel = new BPHChi2Select( probMin );
  par.mass    = mass;
  par.sigma   = sigma;
  par.updated = false;
  return;
}


void BPHOniaToMuMuBuilder::extractList( oniaType type ) {
  if ( !updated ) build();
  OniaParameters& par = oniaPar[type];
  vector<BPHPlusMinusConstCandPtr>& list = oniaList[type];
  if ( par.updated ) return;
  int i;
  int n = fullList.size();
  list.clear();
  list.reserve( n );
  for ( i = 0; i < n; ++i ) {
    BPHPlusMinusConstCandPtr ptr = fullList[i];
    const reco::Candidate* mcPos = ptr->getDaug( "MuPos" );
    const reco::Candidate* mcNeg = ptr->getDaug( "MuNeg" );
    const reco::Candidate* muPos = ptr->originalReco( mcPos );
    const reco::Candidate* muNeg = ptr->originalReco( mcNeg );
    if ( !par.massSel->accept( *ptr   ) ) continue;
    if ( !par.  ptSel->accept( *muPos ) ) continue;
    if ( !par. etaSel->accept( *muPos ) ) continue;
    if ( !par.  ptSel->accept( *muNeg ) ) continue;
    if ( !par. etaSel->accept( *muNeg ) ) continue;
    if ( !par.chi2Sel->accept( *ptr   ) ) continue;
    BPHPlusMinusCandidate* np = new BPHPlusMinusCandidate( evSetup );
    np->add( "MuPos", muPos, ptr->getTrackSearchList( mcPos ),
             BPHParticleMasses::muonMass );
    np->add( "MuNeg", muNeg, ptr->getTrackSearchList( mcNeg ),
             BPHParticleMasses::muonMass );
    if ( par.mass > 0.0 )
    np->setConstraint( par.mass, par.sigma );
    list.push_back( BPHPlusMinusConstCandPtr( np ) );
  }
  par.updated = true;
  return;
}

