#include "HeavyFlavorAnalysis/SpecificDecay/plugins/BPHWriteSpecificDecay.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoSelect.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHMomentumSelect.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHVertexSelect.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHTrackReference.h"

#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHMuonPtSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHMuonEtaSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticlePtSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticleNeutralVeto.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHMultiSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHMassSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHChi2Select.h"

#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHOniaToMuMuBuilder.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHKx0ToKPiBuilder.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHPhiToKKBuilder.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHBuToJPsiKBuilder.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHBsToJPsiPhiBuilder.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHBdToJPsiKxBuilder.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/PatCandidates/interface/GenericParticle.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistance.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <set>
#include <string>
#include <iostream>
#include <fstream>

using namespace std;

#define SET_PAR(TYPE,NAME,PSET) ( NAME = PSET.getParameter< TYPE >( #NAME ) )
// SET_PAR(string,xyz,ps);
// is equivalent to
// ( xyz = ps.getParameter< string >( "xyx" ) )

BPHWriteSpecificDecay::BPHWriteSpecificDecay( const edm::ParameterSet& ps ) {

  usePV = ( SET_PAR( string, pVertexLabel, ps ) != "" );
  usePM = ( SET_PAR( string, patMuonLabel, ps ) != "" );
  useCC = ( SET_PAR( string, ccCandsLabel, ps ) != "" );
  usePF = ( SET_PAR( string, pfCandsLabel, ps ) != "" );
  usePC = ( SET_PAR( string, pcCandsLabel, ps ) != "" );
  useGP = ( SET_PAR( string, gpCandsLabel, ps ) != "" );
  SET_PAR( string, oniaName, ps );
  SET_PAR( string,   sdName, ps );
  SET_PAR( string,   ssName, ps );
  SET_PAR( string,   buName, ps );
  SET_PAR( string,   bdName, ps );
  SET_PAR( string,   bsName, ps );

  SET_PAR( bool, writeMomentum, ps );
  SET_PAR( bool, writeVertex  , ps );

  rMap["Onia"   ] = Onia;
  rMap["PHiMuMu"] = Pmm;
  rMap["Psi1"   ] = Psi1;
  rMap["Psi2"   ] = Psi2;
  rMap["Ups"    ] = Ups;
  rMap["Ups1"   ] = Ups1;
  rMap["Ups2"   ] = Ups2;
  rMap["Ups3"   ] = Ups3;
  rMap["Kx0"    ] = Kx0;
  rMap["PhiKK"  ] = Pkk;
  rMap["Bu"     ] = Bu;
  rMap["Bd"     ] = Bd;
  rMap["Bs"     ] = Bs;

  pMap["ptMin"      ] = ptMin;
  pMap["etaMax"     ] = etaMax;
  pMap["mJPsiMin"   ] = mPsiMin;
  pMap["mJPsiMax"   ] = mPsiMax;
  pMap["mKx0Min"    ] = mKx0Min;
  pMap["mKx0Max"    ] = mKx0Max;
  pMap["mPhiMin"    ] = mPhiMin;
  pMap["mPhiMax"    ] = mPhiMax;
  pMap["massMin"    ] = massMin;
  pMap["massMax"    ] = massMax;
  pMap["probMin"    ] = probMin;
  pMap["massFitMin" ] = mFitMin;
  pMap["massFitMax" ] = mFitMax;
  pMap["constrMass" ] = constrMass;
  pMap["constrSigma"] = constrSigma;

  fMap["constrMJPsi"   ] = constrMJPsi;
  fMap["writeCandidate"] = writeCandidate;

  recoOnia = 
  recoKx0  = writeKx0  =
  recoPkk  = writePkk  =
  recoBu   = writeBu   =
  recoBd   = writeBd   =
  recoBs   = writeBs   = false;

  writeOnia = true;
  const vector<edm::ParameterSet> recoSelect =
        ps.getParameter< vector<edm::ParameterSet> >( "recoSelect" );
  int iSel;
  int nSel = recoSelect.size();
  for ( iSel = 0; iSel < nSel; ++iSel ) setRecoParameters( recoSelect[iSel] );
  if ( !recoOnia ) writeOnia = false;

  if (  recoBu )  recoOnia = true;
  if (  recoBd )  recoOnia =  recoKx0 = true;
  if (  recoBs )  recoOnia =  recoPkk = true;
  if ( writeBu ) writeOnia = true;
  if ( writeBd ) writeOnia = writeKx0 = true;
  if ( writeBs ) writeOnia = writePkk = true;

  if ( usePV ) consume< vector<reco::Vertex                > >( pVertexToken,
                                                                pVertexLabel );
  if ( usePM ) consume< pat::MuonCollection                  >( patMuonToken,
                                                                patMuonLabel );
  if ( useCC ) consume< vector<pat::CompositeCandidate     > >( ccCandsToken,
                                                                ccCandsLabel );
  if ( usePF ) consume< vector<reco::PFCandidate           > >( pfCandsToken,
                                                                pfCandsLabel );
  if ( usePC ) consume< vector<BPHTrackReference::candidate> >( pcCandsToken,
                                                                pcCandsLabel );
  if ( useGP ) consume< vector<pat::GenericParticle        > >( gpCandsToken,
                                                                gpCandsLabel );

  if ( writeOnia ) produces<pat::CompositeCandidateCollection>( oniaName );
  if ( writeKx0  ) produces<pat::CompositeCandidateCollection>(   sdName );
  if ( writePkk  ) produces<pat::CompositeCandidateCollection>(   ssName );
  if ( writeBu   ) produces<pat::CompositeCandidateCollection>(   buName );
  if ( writeBd   ) produces<pat::CompositeCandidateCollection>(   bdName );
  if ( writeBs   ) produces<pat::CompositeCandidateCollection>(   bsName );

}


BPHWriteSpecificDecay::~BPHWriteSpecificDecay() {
}


void BPHWriteSpecificDecay::fillDescriptions(
                            edm::ConfigurationDescriptions& descriptions ) {
   edm::ParameterSetDescription desc;
   desc.add<string>( "pVertexLabel", "" );
   desc.add<string>( "patMuonLabel", "" );
   desc.add<string>( "ccCandsLabel", "" );
   desc.add<string>( "pfCandsLabel", "" );
   desc.add<string>( "pcCandsLabel", "" );
   desc.add<string>( "gpCandsLabel", "" );
   desc.add<string>( "oniaName", "oniaCand" );
   desc.add<string>(   "sdName",  "kx0Cand" );
   desc.add<string>(   "ssName",  "phiCand" );
   desc.add<string>(   "buName", "buFitted" );
   desc.add<string>(   "bdName", "bdFitted" );
   desc.add<string>(   "bsName", "bsFitted" );
   desc.add<bool>( "writeVertex"  , true );
   desc.add<bool>( "writeMomentum", true );
   edm::ParameterSetDescription dpar;
   dpar.add<string>(       "name" );
   dpar.add<double>(       "ptMin", -2.0e35 );
   dpar.add<double>(      "etaMax", -2.0e35 );
   dpar.add<double>(    "mJPsiMin", -2.0e35 );
   dpar.add<double>(    "mJPsiMax", -2.0e35 );
   dpar.add<double>(     "mKx0Min", -2.0e35 );
   dpar.add<double>(     "mKx0Max", -2.0e35 );
   dpar.add<double>(     "mPhiMin", -2.0e35 );
   dpar.add<double>(     "mPhiMax", -2.0e35 );
   dpar.add<double>(     "massMin", -2.0e35 );
   dpar.add<double>(     "massMax", -2.0e35 );
   dpar.add<double>(     "probMin", -2.0e35 );
   dpar.add<double>(  "massFitMin", -2.0e35 );
   dpar.add<double>(  "massFitMax", -2.0e35 );
   dpar.add<double>(  "constrMass", -2.0e35 );
   dpar.add<double>( "constrSigma", -2.0e35 );
   dpar.add<bool>(    "constrMJPsi", true );
   dpar.add<bool>( "writeCandidate", true );
   vector<edm::ParameterSet> rpar;
   desc.addVPSet( "recoSelect", dpar, rpar );
   descriptions.add( "bphWriteSpecificDecay", desc );
   return;
}


void BPHWriteSpecificDecay::beginJob() {
  return;
}


void BPHWriteSpecificDecay::produce( edm::Event& ev,
                                     const edm::EventSetup& es ) {
  fill( ev, es );
  if ( writeOnia ) write( ev, lFull, oniaName );
  if ( writeKx0  ) write( ev, lSd  ,   sdName );
  if ( writePkk  ) write( ev, lSs  ,   ssName );
  if ( writeBu   ) write( ev, lBu  ,   buName );
  if ( writeBd   ) write( ev, lBd  ,   bdName );
  if ( writeBs   ) write( ev, lBs  ,   bsName );
  return;
}


void BPHWriteSpecificDecay::fill( edm::Event& ev,
                                  const edm::EventSetup& es ) {

  lFull.clear();
  lJPsi.clear();
  lSd.clear();
  lSs.clear();
  lBu.clear();
  lBd.clear();
  lBs.clear();
  jPsiOMap.clear();
  pvRefMap.clear();
  ccRefMap.clear();

  // get magnetic field
  edm::ESHandle<MagneticField> magneticField;
  es.get<IdealMagneticFieldRecord>().get( magneticField );

  // get object collections
  // collections are got through "BPHTokenWrapper" interface to allow
  // uniform access in different CMSSW versions

  edm::Handle< std::vector<reco::Vertex> > pVertices;
  pVertexToken.get( ev, pVertices );
  int npv = pVertices->size();

  int nrc = 0;

  // get reco::PFCandidate collection (in full AOD )
  edm::Handle< vector<reco::PFCandidate> > pfCands;
  if ( usePF ) {
    pfCandsToken.get( ev, pfCands );
    nrc = pfCands->size();
  }

  // get pat::PackedCandidate collection (in MiniAOD)
  // pat::PackedCandidate is not defined in CMSSW_5XY, so a
  // typedef (BPHTrackReference::candidate) is used, actually referring 
  // to pat::PackedCandidate only for CMSSW versions where it's defined
  edm::Handle< vector<BPHTrackReference::candidate> > pcCands;
  if ( usePC ) {
    pcCandsToken.get( ev, pcCands );
    nrc = pcCands->size();
  }

  // get pat::GenericParticle collection (in skimmed data)
  edm::Handle< vector<pat::GenericParticle> > gpCands;
  if ( useGP ) {
    gpCandsToken.get( ev, gpCands );
    nrc = gpCands->size();
  }

  // get pat::Muon collection (in full AOD and MiniAOD)
  edm::Handle<pat::MuonCollection> patMuon;
  if ( usePM ) {
    patMuonToken.get( ev, patMuon );
  }

  // get muons from pat::CompositeCandidate objects describing onia;
  // muons from all composite objects are copied to an unique std::vector
  vector<const reco::Candidate*> muDaugs;
  set<const pat::Muon*> muonSet;
  typedef multimap<const reco::Candidate*,
                   const pat::CompositeCandidate*> mu_cc_map;
  mu_cc_map muCCMap;
  if ( useCC ) {
    edm::Handle< vector<pat::CompositeCandidate> > ccCands;
    ccCandsToken.get( ev, ccCands );
    int n = ccCands->size();
    muDaugs.clear();
    muDaugs.reserve( n );
    muonSet.clear();
    set<const pat::Muon*>::const_iterator iter;
    set<const pat::Muon*>::const_iterator iend;
    int i;
    for ( i = 0; i < n; ++i ) {
      const pat::CompositeCandidate& cc = ccCands->at( i );
      int j;
      int m = cc.numberOfDaughters();
      for ( j = 0; j < m; ++j ) {
        const reco::Candidate* dp = cc.daughter( j );
        const pat::Muon* mp = dynamic_cast<const pat::Muon*>( dp );
        iter = muonSet.begin();
        iend = muonSet.end();
        bool add = ( mp != 0 ) && ( muonSet.find( mp ) == iend );
        while ( add && ( iter != iend ) ) {
          if ( BPHRecoBuilder::sameTrack( mp, *iter++, 1.0e-5 ) ) add = false;
        }
        if ( add ) muonSet.insert( mp );
        // associate muon to the CompositeCandidate containing it
        muCCMap.insert( pair<const reco::Candidate*,
                             const pat::CompositeCandidate*>( dp, &cc ) );
      }
    }
    iter = muonSet.begin();
    iend = muonSet.end();
    while ( iter != iend ) muDaugs.push_back( *iter++ );
  }

  map< recoType, map<parType,double> >::const_iterator rIter = parMap.begin();
  map< recoType, map<parType,double> >::const_iterator rIend = parMap.end();

  // reconstruct quarkonia

  BPHOniaToMuMuBuilder* onia = 0;
  if ( recoOnia ) {
    if ( usePM ) onia = new BPHOniaToMuMuBuilder( es,
                        BPHRecoBuilder::createCollection( patMuon, "cfmig" ),
                        BPHRecoBuilder::createCollection( patMuon, "cfmig" ) );
    else
    if ( useCC ) onia = new BPHOniaToMuMuBuilder( es,
                        BPHRecoBuilder::createCollection( muDaugs, "cfmig" ),
                        BPHRecoBuilder::createCollection( muDaugs, "cfmig" ) );
  }

  if ( onia != 0 ) {
    while ( rIter != rIend ) {
      const map< recoType, map<parType,double> >::value_type& rEntry = *rIter++;
      recoType                   rType = rEntry.first;
      const map<parType,double>& pMap  = rEntry.second;
      BPHOniaToMuMuBuilder::oniaType type;
      switch( rType ) {
      case Pmm : type = BPHOniaToMuMuBuilder::Phi ; break;
      case Psi1: type = BPHOniaToMuMuBuilder::Psi1; break;
      case Psi2: type = BPHOniaToMuMuBuilder::Psi2; break;
      case Ups : type = BPHOniaToMuMuBuilder::Ups ; break;
      case Ups1: type = BPHOniaToMuMuBuilder::Ups1; break;
      case Ups2: type = BPHOniaToMuMuBuilder::Ups2; break;
      case Ups3: type = BPHOniaToMuMuBuilder::Ups3; break;
      default: continue;
      }
      map<parType,double>::const_iterator pIter = pMap.begin();
      map<parType,double>::const_iterator pIend = pMap.end();
      while ( pIter != pIend ) {
        const map<parType,double>::value_type& pEntry = *pIter++;
        parType id = pEntry.first;
        double  pv = pEntry.second;
        switch( id ) {
        case ptMin      : onia->setPtMin  ( type, pv ); break;
        case etaMax     : onia->setEtaMax ( type, pv ); break;
        case massMin    : onia->setMassMin( type, pv ); break;
        case massMax    : onia->setMassMax( type, pv ); break;
        case probMin    : onia->setProbMin( type, pv ); break;
        case constrMass : onia->setConstr ( type, pv, 
                                                  onia->getConstrSigma( type )
                                                     ); break;
        case constrSigma: onia->setConstr ( type, onia->getConstrMass ( type ),
                                                  pv ); break;
        default: break;
        }
      }
    }
    lFull = onia->build();
  }

  // associate onia to primary vertex

  int iFull;
  int nFull = lFull.size();
  map<const BPHRecoCandidate*,const reco::Vertex*> oniaVtxMap;

  typedef mu_cc_map::const_iterator mu_cc_iter;
  for ( iFull = 0; iFull < nFull; ++iFull ) {

    const reco::Vertex* pVtx = 0;
    int pvId = 0;
    const BPHPlusMinusCandidate* ptr = lFull[iFull].get();
    const std::vector<const reco::Candidate*>& daugs = ptr->daughters();

    // try to recover primary vertex association in skim data:
    // get the CompositeCandidate containing both muons
    pair<mu_cc_iter,mu_cc_iter> cc0 = muCCMap.equal_range(
                                      ptr->originalReco( daugs[0] ) );
    pair<mu_cc_iter,mu_cc_iter> cc1 = muCCMap.equal_range(
                                      ptr->originalReco( daugs[1] ) );
    mu_cc_iter iter0 = cc0.first;
    mu_cc_iter iend0 = cc0.second;
    mu_cc_iter iter1 = cc1.first;
    mu_cc_iter iend1 = cc1.second;
    while ( ( iter0 != iend0 ) && ( pVtx == 0 )  ) {
      const pat::CompositeCandidate* ccp = iter0++->second;
      while ( iter1 != iend1 ) {
        if ( ccp != iter1++->second ) continue;
        pVtx = ccp->userData<reco::Vertex>( "PVwithmuons" );
        const reco::Vertex* sVtx = 0;
        const reco::Vertex::Point& pPos = pVtx->position();
        float dMin = 999999.;
        int ipv;
        for ( ipv = 0; ipv < npv; ++ipv ) {
          const reco::Vertex* tVtx = &pVertices->at( ipv );
          const reco::Vertex::Point& tPos = tVtx->position();
          float dist = pow( pPos.x() - tPos.x(), 2 ) +
                       pow( pPos.y() - tPos.y(), 2 ) +
                       pow( pPos.z() - tPos.z(), 2 );
          if ( dist < dMin ) {
            dMin = dist;    
            sVtx = tVtx;
            pvId = ipv;
          }
        }
        pVtx = sVtx;
        break;
      }
    }

    // if not found, as ofr other type of inut data, 
    // try to get the nearest primary vertex in z direction
    if ( pVtx == 0 ) {
      const reco::Vertex::Point& sVtp = ptr->vertex().position();
      GlobalPoint  cPos( sVtp.x(), sVtp.y(), sVtp.z() );
      const pat::CompositeCandidate& sCC = ptr->composite();
      GlobalVector cDir( sCC.px(), sCC.py(), sCC.pz() );
      GlobalPoint  bPos( 0.0, 0.0, 0.0 );
      GlobalVector bDir( 0.0, 0.0, 1.0 );
      TwoTrackMinimumDistance ttmd;
      bool state = ttmd.calculate( GlobalTrajectoryParameters( cPos, cDir,
                                   TrackCharge( 0 ), &( *magneticField ) ),
                                   GlobalTrajectoryParameters( bPos, bDir,
                                   TrackCharge( 0 ), &( *magneticField ) ) );
      float minDz = 999999.;
      float extrapZ = ( state ? ttmd.points().first.z() : -9e20 );
      int ipv;
      for ( ipv = 0; ipv < npv; ++ipv ) {
        const reco::Vertex& tVtx = pVertices->at( ipv );
        float deltaZ = fabs( extrapZ - tVtx.position().z() ) ;
        if ( deltaZ < minDz ) {
          minDz = deltaZ;    
          pVtx = &tVtx;
          pvId = ipv;
        }
      }
    }

    oniaVtxMap[ptr] = pVtx;
    pvRefMap[ptr] = vertex_ref( pVertices, pvId );

  }
  pVertexToken.get( ev, pVertices );

  // get JPsi subsample and associate JPsi candidate to original 
  // generic onia candidate
  if ( nFull ) lJPsi = onia->getList( BPHOniaToMuMuBuilder::Psi1 );

  int nJPsi = lJPsi.size();
  delete onia;

  if ( !nJPsi ) return;
  if ( !nrc   ) return;

  int ij;
  int io;
  int nj = lJPsi.size();
  int no = lFull.size();
  for ( ij = 0; ij < nj; ++ij ) {
    const BPHRecoCandidate* jp = lJPsi[ij].get();
    for ( io = 0; io < no; ++io ) {
      const BPHRecoCandidate* oc = lFull[io].get();
      if ( ( jp->originalReco( jp->getDaug( "MuPos" ) ) ==
             oc->originalReco( oc->getDaug( "MuPos" ) ) ) &&
           ( jp->originalReco( jp->getDaug( "MuNeg" ) ) ==
             oc->originalReco( oc->getDaug( "MuNeg" ) ) ) ) {
        jPsiOMap[jp] = oc;
        break;
      }
    }
  }

  // build and dump Bu

  BPHBuToJPsiKBuilder* bu = 0;
  if ( recoBu ) {
    if ( usePF ) bu = new BPHBuToJPsiKBuilder( es, lJPsi,
                          BPHRecoBuilder::createCollection( pfCands ) );
    else
    if ( usePC ) bu = new BPHBuToJPsiKBuilder( es, lJPsi,
                          BPHRecoBuilder::createCollection( pcCands ) );
    else
    if ( useGP ) bu = new BPHBuToJPsiKBuilder( es, lJPsi,
                          BPHRecoBuilder::createCollection( gpCands ) );
  }

  if ( bu != 0 ) {
    rIter = parMap.find( Bu );
    if ( rIter != rIend ) {
      const map<parType,double>& pMap = rIter->second;
      map<parType,double>::const_iterator pIter = pMap.begin();
      map<parType,double>::const_iterator pIend = pMap.end();
      while ( pIter != pIend ) {
        const map<parType,double>::value_type& pEntry = *pIter++;
        parType id = pEntry.first;
        double  pv = pEntry.second;
        switch( id ) {
        case ptMin      : bu->setKPtMin     ( pv ); break;
        case etaMax     : bu->setKEtaMax    ( pv ); break;
        case mPsiMin    : bu->setJPsiMassMin( pv ); break;
        case mPsiMax    : bu->setJPsiMassMax( pv ); break;
        case massMin    : bu->setMassMin    ( pv ); break;
        case massMax    : bu->setMassMax    ( pv ); break;
        case probMin    : bu->setProbMin    ( pv ); break;
        case mFitMin    : bu->setMassFitMin ( pv ); break;
        case mFitMax    : bu->setMassFitMax ( pv ); break;
        case constrMJPsi: bu->setConstr     ( pv > 0 ); break;
        case writeCandidate: writeBu =      ( pv > 0 ); break;
        default: break;
        }
      }
    }
    lBu = bu->build();
    delete bu;
  }

  // build and dump Kx0

  vector<BPHPlusMinusConstCandPtr> lKx0;
  BPHKx0ToKPiBuilder* kx0 = 0;
  if ( recoKx0 ) {
    if ( usePF ) kx0 = new BPHKx0ToKPiBuilder( es,
                       BPHRecoBuilder::createCollection( pfCands ),
                       BPHRecoBuilder::createCollection( pfCands ) );
    else
    if ( usePC ) kx0 = new BPHKx0ToKPiBuilder( es,
                       BPHRecoBuilder::createCollection( pcCands ),
                       BPHRecoBuilder::createCollection( pcCands ) );
    else
    if ( useGP ) kx0 = new BPHKx0ToKPiBuilder( es,
                       BPHRecoBuilder::createCollection( gpCands ),
                       BPHRecoBuilder::createCollection( gpCands ) );
  }

  if ( kx0 != 0 ) {
    rIter = parMap.find( Kx0 );
    if ( rIter != rIend ) {
      const map<parType,double>& pMap = rIter->second;
      map<parType,double>::const_iterator pIter = pMap.begin();
      map<parType,double>::const_iterator pIend = pMap.end();
      while ( pIter != pIend ) {
        const map<parType,double>::value_type& pEntry = *pIter++;
        parType id = pEntry.first;
        double  pv = pEntry.second;
        switch( id ) {
        case ptMin      : kx0->setPtMin  ( pv ); break;
        case etaMax     : kx0->setEtaMax ( pv ); break;
        case massMin    : kx0->setMassMin( pv ); break;
        case massMax    : kx0->setMassMax( pv ); break;
        case probMin    : kx0->setProbMin( pv ); break;
        case constrMass : kx0->setConstr ( pv, kx0->getConstrSigma() ); break;
        case constrSigma: kx0->setConstr ( kx0->getConstrMass() , pv ); break;
        case writeCandidate: writeKx0 =  ( pv > 0 ); break;
        default: break;
        }
      }
    }
    lKx0 = kx0->build();
    delete kx0;
  }

  int nKx0 = lKx0.size();

  // build and dump Bd

  if ( recoBd && nKx0 ) {

    BPHBdToJPsiKxBuilder* bd = new BPHBdToJPsiKxBuilder( es, lJPsi, lKx0 );
    rIter = parMap.find( Bd );
    if ( rIter != rIend ) {
      const map<parType,double>& pMap = rIter->second;
      map<parType,double>::const_iterator pIter = pMap.begin();
      map<parType,double>::const_iterator pIend = pMap.end();
      while ( pIter != pIend ) {
        const map<parType,double>::value_type& pEntry = *pIter++;
        parType id = pEntry.first;
        double  pv = pEntry.second;
        switch( id ) {
        case mPsiMin    : bd->setJPsiMassMin( pv ); break;
        case mPsiMax    : bd->setJPsiMassMax( pv ); break;
        case mKx0Min    : bd->setKxMassMin  ( pv ); break;
        case mKx0Max    : bd->setKxMassMax  ( pv ); break;
        case massMin    : bd->setMassMin    ( pv ); break;
        case massMax    : bd->setMassMax    ( pv ); break;
        case probMin    : bd->setProbMin    ( pv ); break;
        case mFitMin    : bd->setMassFitMin ( pv ); break;
        case mFitMax    : bd->setMassFitMax ( pv ); break;
        case constrMJPsi: bd->setConstr     ( pv > 0 ); break;
        case writeCandidate: writeBd =      ( pv > 0 ); break;
        default: break;
        }
      }
    }

    lBd = bd->build();
    delete bd;

    set<BPHRecoConstCandPtr> sKx0;
    int iBd;
    int nBd = lBd.size();
    for ( iBd = 0; iBd < nBd; ++iBd ) sKx0.insert( lBd[iBd]->getComp( "Kx0" ) );
    set<BPHRecoConstCandPtr>::const_iterator iter = sKx0.begin();
    set<BPHRecoConstCandPtr>::const_iterator iend = sKx0.end();
    while ( iter != iend ) lSd.push_back( *iter++ );

  }

  // build and dump Phi

  vector<BPHPlusMinusConstCandPtr> lPhi;
  BPHPhiToKKBuilder* phi = 0;
  if ( recoPkk ) {
    if ( usePF ) phi = new BPHPhiToKKBuilder( es,
                       BPHRecoBuilder::createCollection( pfCands ),
                       BPHRecoBuilder::createCollection( pfCands ) );
    else
    if ( usePC ) phi = new BPHPhiToKKBuilder( es,
                       BPHRecoBuilder::createCollection( pcCands ),
                       BPHRecoBuilder::createCollection( pcCands ) );
    else
    if ( useGP ) phi = new BPHPhiToKKBuilder( es,
                       BPHRecoBuilder::createCollection( gpCands ),
                       BPHRecoBuilder::createCollection( gpCands ) );
  }

  if ( phi != 0 ) {
    rIter = parMap.find( Pkk );
    if ( rIter != rIend ) {
      const map<parType,double>& pMap = rIter->second;
      map<parType,double>::const_iterator pIter = pMap.begin();
      map<parType,double>::const_iterator pIend = pMap.end();
      while ( pIter != pIend ) {
        const map<parType,double>::value_type& pEntry = *pIter++;
        parType id = pEntry.first;
        double  pv = pEntry.second;
        switch( id ) {
        case ptMin      : phi->setPtMin  ( pv ); break;
        case etaMax     : phi->setEtaMax ( pv ); break;
        case massMin    : phi->setMassMin( pv ); break;
        case massMax    : phi->setMassMax( pv ); break;
        case probMin    : phi->setProbMin( pv ); break;
        case constrMass : phi->setConstr ( pv, phi->getConstrSigma() ); break;
        case constrSigma: phi->setConstr ( phi->getConstrMass() , pv ); break;
        case writeCandidate: writePkk =  ( pv > 0 ); break;
        default: break;
        }
      }
    }
    lPhi = phi->build();
    delete phi;
  }

  int nPhi = lPhi.size();

  // build and dump Bs

  if ( recoBs && nPhi ) {

    BPHBsToJPsiPhiBuilder* bs = new BPHBsToJPsiPhiBuilder( es, lJPsi, lPhi );
    rIter = parMap.find( Bs );
    if ( rIter != rIend ) {
      const map<parType,double>& pMap = rIter->second;
      map<parType,double>::const_iterator pIter = pMap.begin();
      map<parType,double>::const_iterator pIend = pMap.end();
      while ( pIter != pIend ) {
        const map<parType,double>::value_type& pEntry = *pIter++;
        parType id = pEntry.first;
        double  pv = pEntry.second;
        switch( id ) {
        case mPsiMin    : bs->setJPsiMassMin( pv ); break;
        case mPsiMax    : bs->setJPsiMassMax( pv ); break;
        case mPhiMin    : bs->setPhiMassMin ( pv ); break;
        case mPhiMax    : bs->setPhiMassMax ( pv ); break;
        case massMin    : bs->setMassMin    ( pv ); break;
        case massMax    : bs->setMassMax    ( pv ); break;
        case probMin    : bs->setProbMin    ( pv ); break;
        case mFitMin    : bs->setMassFitMin ( pv ); break;
        case mFitMax    : bs->setMassFitMax ( pv ); break;
        case constrMJPsi: bs->setConstr     ( pv > 0 ); break;
        case writeCandidate: writeBs =      ( pv > 0 ); break;
        default: break;
        }
      }
    }

    lBs = bs->build();
    delete bs;

    set<BPHRecoConstCandPtr> sPhi;
    int iBs;
    int nBs = lBs.size();
    for ( iBs = 0; iBs < nBs; ++iBs ) sPhi.insert( lBs[iBs]->getComp( "Phi" ) );
    set<BPHRecoConstCandPtr>::const_iterator iter = sPhi.begin();
    set<BPHRecoConstCandPtr>::const_iterator iend = sPhi.end();
    while ( iter != iend ) lSs.push_back( *iter++ );

  }

  return;

}


void BPHWriteSpecificDecay::endJob() {
  return;
}


void BPHWriteSpecificDecay::setRecoParameters( const edm::ParameterSet& ps ) {

  const string& name = ps.getParameter<string>( "name" );
  bool writeCandidate = ps.getParameter<bool>( "writeCandidate" );
  switch( rMap[name] ) {
  case Onia: recoOnia = true; writeOnia = writeCandidate; break;
  case Pmm:
  case Psi1:
  case Psi2:
  case Ups:
  case Ups1:
  case Ups2:
  case Ups3: recoOnia = true;                             break;
  case Kx0 : recoKx0  = true; writeKx0  = writeCandidate; break;
  case Pkk : recoPkk  = true; writePkk  = writeCandidate; break;
  case Bu  : recoBu   = true; writeBu   = writeCandidate; break;
  case Bd  : recoBd   = true; writeBd   = writeCandidate; break;
  case Bs  : recoBs   = true; writeBs   = writeCandidate; break;
  }

  map<string,parType>::const_iterator pIter = pMap.begin();
  map<string,parType>::const_iterator pIend = pMap.end();
  while ( pIter != pIend ) {
    const map<string,parType>::value_type& entry = *pIter++;
    const string& pn = entry.first;
    parType       id = entry.second;
    double pv = ps.getParameter<double>( pn );
    if ( pv > -1.0e35 ) edm::LogVerbatim( "Configuration" )
         << "BPHWriteSpecificDecay::setRecoParameters: set " << pn
         << " for " << name << " : "
         << ( parMap[rMap[name]][id] = pv );
  }

  map<string,parType>::const_iterator fIter = fMap.begin();
  map<string,parType>::const_iterator fIend = fMap.end();
  while ( fIter != fIend ) {
    const map<string,parType>::value_type& entry = *fIter++;
    const string& fn = entry.first;
    parType       id = entry.second;
    edm::LogVerbatim( "Configuration" )
         << "BPHWriteSpecificDecay::setRecoParameters: set " << fn
         << " for " << name << " : "
         << ( parMap[rMap[name]][id] =
                                     ( ps.getParameter<bool>( fn ) ? 1 : -1 ) );
  }

}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( BPHWriteSpecificDecay );
