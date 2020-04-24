/*
 *  See header file for a description of this class.
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHKinematicFit.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicParticleFactoryFromTransientTrack.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicParticle.h"
#include "RecoVertex/KinematicFit/interface/KinematicParticleVertexFitter.h"
#include "RecoVertex/KinematicFit/interface/KinematicConstrainedVertexFitter.h"
#include "RecoVertex/KinematicFit/interface/KinematicParticleFitter.h"
#include "RecoVertex/KinematicFit/interface/MassKinematicConstraint.h"
#include "RecoVertex/KinematicFit/interface/TwoTrackMassKinematicConstraint.h"
#include "RecoVertex/KinematicFit/interface/MultiTrackMassKinematicConstraint.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
BPHKinematicFit::BPHKinematicFit():
 BPHDecayVertex( 0 ),
 massConst( -1.0 ),
 massSigma( -1.0 ),
 oldKPs( true ),
 oldFit( true ),
 oldMom( true ),
 kinTree( 0 ) {
}


BPHKinematicFit::BPHKinematicFit( const BPHKinematicFit* ptr ):
 BPHDecayVertex( ptr, 0 ),
 massConst( -1.0 ),
 massSigma( -1.0 ),
 oldKPs( true ),
 oldFit( true ),
 oldMom( true ),
 kinTree( 0 ) {
  map<const reco::Candidate*,const reco::Candidate*> iMap;
  const vector<const reco::Candidate*>& daug = daughters();
  const vector<Component>& list = ptr->componentList();
  int i;
  int n = daug.size();
  for ( i = 0; i < n; ++i ) {
    const reco::Candidate* cand = daug[i];
    iMap[originalReco( cand )] = cand;
  }
  for ( i = 0; i < n; ++i ) {
    const Component& c = list[i];
    dMSig[iMap[c.cand]] = c.msig;
  }
  const vector<BPHRecoConstCandPtr>& dComp = daughComp();
  int j;
  int m = dComp.size();
  for ( j = 0; j < m; ++j ) {
    const map<const reco::Candidate*,double>& dMap = dComp[j]->dMSig;
    dMSig.insert( dMap.begin(), dMap.end() );
  }
}

//--------------
// Destructor --
//--------------
BPHKinematicFit::~BPHKinematicFit() {
}

//--------------
// Operations --
//--------------
void BPHKinematicFit::setConstraint( double mass, double sigma ) {
  oldFit = oldMom = true;
  massConst = mass;
  massSigma = sigma;
  return;
}


double BPHKinematicFit::constrMass() const {
  return massConst;
}


double BPHKinematicFit::constrSigma() const {
  return massSigma;
}


const vector<RefCountedKinematicParticle>& BPHKinematicFit::kinParticles()
                                                            const {
  if ( oldKPs ) buildParticles();
  return allParticles;
}


vector<RefCountedKinematicParticle> BPHKinematicFit::kinParticles(
                                    const vector<string>& names ) const {
  if ( oldKPs ) buildParticles();
  const vector<const reco::Candidate*>& daugs = daughFull();
  vector<RefCountedKinematicParticle> plist;
  if ( allParticles.size() != daugs.size() ) return plist;
  set<RefCountedKinematicParticle> pset;
  int i;
  int n = names.size();
  int m = daugs.size();
  plist.reserve( m );
  for ( i = 0; i < n; ++i ) {
    const string& pname = names[i];
    if ( pname == "*" ) {
      int j = m;
      while ( j-- ) {
        RefCountedKinematicParticle& kp = allParticles[j];
        if ( pset.find( kp ) != pset.end() ) continue;
        plist.push_back( kp );
        pset .insert   ( kp );
      }
      break;
    }
    map<const reco::Candidate*,
        RefCountedKinematicParticle>::const_iterator iter = kinMap.find(
                                                            getDaug( pname ) );
    map<const reco::Candidate*,
        RefCountedKinematicParticle>::const_iterator iend = kinMap.end();
    if ( iter != iend ) {
      const RefCountedKinematicParticle& kp = iter->second;
      if ( pset.find( kp ) != pset.end() ) continue;
      plist.push_back( kp );
      pset .insert   ( kp );
    }
    else {
      edm::LogPrint( "ParticleNotFound" )
                  << "BPHKinematicFit::kinParticles: "
                  << pname << " not found";
    }
  }
  return plist;
}


const RefCountedKinematicTree& BPHKinematicFit::kinematicTree() const {
  if ( oldFit ) return kinematicTree( "", massConst, massSigma );
  return kinTree;
}


const RefCountedKinematicTree& BPHKinematicFit::kinematicTree(
                               const string& name,
                               double mass, double sigma ) const {
  if ( sigma < 0 ) return kinematicTree( name, mass );
  ParticleMass mc = mass;
  MassKinematicConstraint kinConst( mc, sigma );
  return kinematicTree( name, &kinConst );
}


const RefCountedKinematicTree& BPHKinematicFit::kinematicTree(
                               const string& name,
                               double mass ) const {
  if ( mass < 0 ) {
    kinTree = RefCountedKinematicTree( 0 );
    oldFit = false;
    return kinTree;
  }
  int nn = daughFull().size();
  ParticleMass mc = mass;
  if ( nn == 2 ) {
    TwoTrackMassKinematicConstraint   kinConst( mc );
    return kinematicTree( name, &kinConst );
  }
  else {
    MultiTrackMassKinematicConstraint kinConst( mc, nn );
    return kinematicTree( name, &kinConst );
  }
}


const RefCountedKinematicTree& BPHKinematicFit::kinematicTree(
                               const string& name,
                               KinematicConstraint* kc ) const {
  kinTree = RefCountedKinematicTree( 0 );
  oldFit = false;
  kinParticles();
  if ( allParticles.size() != daughFull().size() ) return kinTree;
  vector<RefCountedKinematicParticle> kComp;
  vector<RefCountedKinematicParticle> kTail;
  if ( name != "" ) {
    const BPHRecoCandidate* comp = getComp( name ).get();
    if ( comp == 0 ) {
      edm::LogPrint( "ParticleNotFound" )
                  << "BPHKinematicFit::kinematicTree: "
                  << name << " daughter not found";
      return kinTree;
    }
    const vector<string>& names = comp->daugNames();
    int ns;
    int nn = ns = names.size();
    vector<string> nfull( nn + 1 );
    nfull[nn] = "*";
    while ( nn-- ) nfull[nn] = name + "/" + names[nn];
    vector<RefCountedKinematicParticle> kPart = kinParticles( nfull );
    vector<RefCountedKinematicParticle>::const_iterator iter = kPart.begin();
    vector<RefCountedKinematicParticle>::const_iterator imid = iter + ns;
    vector<RefCountedKinematicParticle>::const_iterator iend = kPart.end();
    kComp.insert( kComp.end(), iter, imid );
    kTail.insert( kTail.end(), imid, iend );
  }
  else {
    kComp = allParticles;
  }
  try {
    KinematicParticleVertexFitter vtxFitter;
    RefCountedKinematicTree compTree = vtxFitter.fit( kComp );
    if ( compTree->isEmpty() ) return kinTree;
    KinematicParticleFitter kinFitter;
    compTree = kinFitter.fit( kc, compTree );
    if ( compTree->isEmpty() ) return kinTree;
    compTree->movePointerToTheTop();
    if ( kTail.size() ) {
      RefCountedKinematicParticle compPart = compTree->currentParticle();
      if ( !compPart->currentState().isValid() ) return kinTree;
      kTail.push_back( compPart );
      kinTree = vtxFitter.fit( kTail );
    }
    else {
      kinTree = compTree;
    }
  }
  catch ( std::exception e ) {
    edm::LogPrint( "FitFailed" )
                << "BPHKinematicFit::kinematicTree: "
                << "kin fit reset";
    kinTree = RefCountedKinematicTree( 0 );
  }
  return kinTree;
}


const RefCountedKinematicTree& BPHKinematicFit::kinematicTree(
                               const string& name,
                               MultiTrackKinematicConstraint* kc ) const {
  kinTree = RefCountedKinematicTree( 0 );
  oldFit = false;
  kinParticles();
  if ( allParticles.size() != daughFull().size() ) return kinTree;
  vector<string> nfull;
  if ( name != "" ) {
    const BPHRecoCandidate* comp = getComp( name ).get();
    if ( comp == 0 ) {
      edm::LogPrint( "ParticleNotFound" )
                  << "BPHKinematicFit::kinematicTree: "
                  << name << " daughter not found";
      return kinTree;
    }
    const vector<string>& names = comp->daugNames();
    int nn = names.size();
    nfull.resize( nn + 1 );
    nfull[nn] = "*";
    while ( nn-- ) nfull[nn] = name + "/" + names[nn];
  }
  else {
    nfull.push_back( "*" );
  }
  try {
    KinematicConstrainedVertexFitter cvf;
    kinTree = cvf.fit( kinParticles( nfull ), kc );
  }
  catch ( std::exception e ) {
    edm::LogPrint( "FitFailed" )
                << "BPHKinematicFit::kinematicTree: "
                << "kin fit reset";
    kinTree = RefCountedKinematicTree( 0 );
  }
  return kinTree;
}


void BPHKinematicFit::resetKinematicFit() const {
  oldKPs = oldFit = oldMom = true;
  return;
}


bool BPHKinematicFit::isEmpty() const {
  kinematicTree();
  if ( kinTree.get() == 0 ) return true;
  return kinTree->isEmpty();
}


bool BPHKinematicFit::isValidFit() const {
  const RefCountedKinematicParticle kPart = currentParticle();
  if ( kPart.get() == 0 ) return false;
  return kPart->currentState().isValid();
}


const RefCountedKinematicParticle BPHKinematicFit::currentParticle() const {
  if ( isEmpty() ) return RefCountedKinematicParticle( 0 );
  return kinTree->currentParticle();
}


const RefCountedKinematicVertex BPHKinematicFit::currentDecayVertex() const {
  if ( isEmpty() ) return RefCountedKinematicVertex( 0 );
  return kinTree->currentDecayVertex();
}


ParticleMass BPHKinematicFit::mass() const {
  const RefCountedKinematicParticle kPart = currentParticle();
  if ( kPart.get() == 0 ) return -1.0;
  const KinematicState kStat = kPart->currentState();
  if ( kStat.isValid() ) return kStat.mass();
  return -1.0;
}


const math::XYZTLorentzVector& BPHKinematicFit::p4() const {
  if ( oldMom ) fitMomentum();
  return totalMomentum;
}


void BPHKinematicFit::addK( const string& name,
                            const reco::Candidate* daug, 
                            double mass, double sigma ) {
  addK( name, daug, "cfhpmig", mass, sigma );
  return;
}


void BPHKinematicFit::addK( const string& name,
                            const reco::Candidate* daug, 
                            const string& searchList,
                            double mass, double sigma ) {
  addV( name, daug, searchList, mass );
  dMSig[daughters().back()] = sigma;
  return;
}


void BPHKinematicFit::addK( const string& name,
                            const BPHRecoConstCandPtr& comp ) {
  addV( name, comp );
  const map<const reco::Candidate*,double>& dMap = comp->dMSig;
  dMSig.insert( dMap.begin(), dMap.end() );
  return;
}


void BPHKinematicFit::setNotUpdated() const {
  BPHDecayVertex::setNotUpdated();
  resetKinematicFit();
  return;
}


void BPHKinematicFit::buildParticles() const {
  kinMap.clear();
  allParticles.clear();
  const vector<const reco::Candidate*>& daug = daughFull();
  KinematicParticleFactoryFromTransientTrack pFactory;
  int n = daug.size();
  allParticles.reserve( n );
  float chi = 0.0;
  float ndf = 0.0;
  while ( n-- ) {
    const reco::Candidate* cand = daug[n];
    ParticleMass mass = cand->mass();
    float sigma = dMSig.find( cand )->second;
    if ( sigma < 0 ) sigma = 1.0e-7;
    reco::TransientTrack* tt = getTransientTrack( cand );
    if ( tt != 0 ) allParticles.push_back( kinMap[cand] =
                                           pFactory.particle( *tt, 
                                           mass, chi, ndf, sigma ) );
  }
  oldKPs = false;
  return;
}


void BPHKinematicFit::fitMomentum() const {
  if ( isValidFit() ) {
    const KinematicState& ks = currentParticle()->currentState();
    GlobalVector tm = ks.globalMomentum();
    double x = tm.x();
    double y = tm.y();
    double z = tm.z();
    double m = ks.mass();
    double e = sqrt( ( x * x ) + ( y * y ) + ( z * z ) + ( m * m ) );
    totalMomentum.SetPxPyPzE( x, y, z, e );
  }
  else {
    edm::LogPrint( "FitNotFound" )
                << "BPHKinematicFit::fitMomentum: "
                << "simple momentum sum computed";
    math::XYZTLorentzVector tm;
    const vector<const reco::Candidate*>& daug = daughters();
    int n = daug.size();
    while ( n-- ) tm += daug[n]->p4();
    const vector<BPHRecoConstCandPtr>& comp = daughComp();
    int m = comp.size();
    while ( m-- ) tm += comp[m]->p4();
    totalMomentum = tm;
  }
  oldMom = false;
  return;
}

