#include "HeavyFlavorAnalysis/SpecificDecay/plugins/BPHHistoSpecificDecay.h"

#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticleMasses.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/PatCandidates/interface/Muon.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include "TMath.h"
#include "Math/VectorUtil.h"
#include "TVector3.h"

#include <TH1.h>
#include <TFile.h>

#include <set>
#include <string>
#include <iostream>
#include <fstream>
#include <math.h>

using namespace std;

#define SET_LABEL(NAME,PSET) ( NAME = PSET.getParameter<string>( #NAME ) )
// SET_LABEL(xyz,ps);
// is equivalent to
// xyz = ps.getParameter<string>( "xyx" )

class BPHUserData {
 public:
  template<class T>
  static const T* get( const pat::CompositeCandidate& cand,
                       const string& name ) {
    if ( cand.hasUserData( name ) ) return cand.userData<T>( name );
    return 0;
  }
  template<class T>
  static const T* getByRef( const pat::CompositeCandidate& cand,
                            const string& name ) {
    if ( cand.hasUserData( name ) ) {
      typedef edm::Ref< std::vector<T> > objRef;
      const objRef* ref = cand.userData<objRef>( name );
      if ( ref ==      0 ) return 0;
      if ( ref->isNull() ) return 0;
      return ref->get();
    }
    return 0;
  }
};


class BPHDaughters {
 public:
  static vector<const reco::Candidate*> get(
         const pat::CompositeCandidate& cand, float massMin, float massMax ) {
    int i;
    int n = cand.numberOfDaughters();
    vector<const reco::Candidate*> v;
    v.reserve( n );
    for ( i = 0; i < n; ++i ) {
      const reco::Candidate* dptr = cand.daughter( i );
      float mass = dptr->mass();
      if ( ( mass > massMin ) && ( mass < massMax ) ) v.push_back( dptr );
    }
    return v;
  }
};


class BPHSoftMuonSelect {
 public:
  BPHSoftMuonSelect( int   cutTrackerLayers = 5,
                     int   cutPixelLayers   = 0,
                     float maxDxy           = 0.3,
                     float maxDz            = 20.0,
                     bool  goodMuon         = true ,
                     bool  highPurity       = true ):
   cutTL ( cutTrackerLayers ),
   cutPL ( cutPixelLayers   ),
   maxXY ( maxDxy           ),
   maxZ  ( maxDz            ),
   gM    ( goodMuon         ),
   hP    ( highPurity       ) {}
  bool accept( const reco::Candidate& cand,
               const reco::Vertex* pv ) const {
    const pat::Muon* p = reinterpret_cast<const pat::Muon*>( &cand );
    if ( p == 0 ) return false;
    if ( gM &&
        !muon::isGoodMuon( *p, muon::TMOneStationTight ) ) return false;
    if ( p->innerTrack()->hitPattern().trackerLayersWithMeasurement() <= cutTL )
         return false;
    if ( p->innerTrack()->hitPattern().  pixelLayersWithMeasurement() <= cutPL )
         return false;
    if ( hP &&
        !p->innerTrack()->quality( reco::TrackBase::highPurity ) )
         return false;
    if ( pv == 0 ) return true;
    const reco::Vertex::Point& pos = pv->position();
    if ( fabs( p->innerTrack()->dxy( pos ) ) >= maxXY )
         return false;
    if ( fabs( p->innerTrack()->dz ( pos ) ) >= maxZ )
         return false;
    return true;
  }
 private:
  const reco::Vertex* pv;
  int cutTL;
  int cutPL;
  float maxXY;
  float maxZ;
  bool gM;
  bool hP;
};


class BPHDaughterSelect: public BPHHistoSpecificDecay::CandidateSelect {
 public:
  BPHDaughterSelect( float  ptMinLoose,
                     float  ptMinTight,
                     float etaMaxLoose,
                     float etaMaxTight,
                     const BPHSoftMuonSelect*
                           softMuonselector = 0 ): pLMin(  ptMinLoose ),
                                                   pTMin(  ptMinTight ),
                                                   eLMax( etaMaxLoose ),
                                                   eTMax( etaMaxTight ),
                                                   sms( softMuonselector ) {
  }
  bool accept( const pat::CompositeCandidate& cand,
               const reco::Vertex* pv = 0 ) const {
    const reco::Candidate* dptr0 = cand.daughter( 0 );
    const reco::Candidate* dptr1 = cand.daughter( 1 );
    if ( dptr0 == 0 ) return false;
    if ( dptr1 == 0 ) return false;
    float pt0 = dptr0->pt();
    float pt1 = dptr1->pt();
    if ( ( pt0 < pLMin ) || ( pt1 < pLMin ) ) return false;
    if ( ( pt0 < pTMin ) && ( pt1 < pTMin ) ) return false;
    float eta0 = fabs( dptr0->eta() );
    float eta1 = fabs( dptr1->eta() );
    if (   ( eLMax > 0 ) &&
         ( ( eta0 > eLMax ) || ( eta1 > eLMax ) ) ) return false;
    if (   ( eTMax > 0 ) && 
         ( ( eta0 > eTMax ) && ( eta1 > eTMax ) ) ) return false;
    if ( sms != 0 ) {
      const reco::Vertex* pvtx = BPHUserData::getByRef
           <reco::Vertex>( cand, "primaryVertex" );
      if ( pvtx == 0 ) return false;
      if ( !sms->accept( *dptr0, pvtx ) ) return false;
      if ( !sms->accept( *dptr1, pvtx ) ) return false;
    }    
    return true;
  }
 private:
  float pLMin;
  float pTMin;
  float eLMax;
  float eTMax;
  const BPHSoftMuonSelect* sms;
};


class BPHCompositeBasicSelect: public BPHHistoSpecificDecay::CandidateSelect {
 public:
  BPHCompositeBasicSelect( float     massMin,
                           float     massMax,
                           float       ptMin = -1.0,
                           float      etaMax = -1.0,
                           float rapidityMax = -1.0 ): mMin(     massMin ),
                                                       mMax(     massMax ),
                                                       pMin(       ptMin ),
                                                       eMax(      etaMax ),
                                                       yMax( rapidityMax ) {
  }
  bool accept( const pat::CompositeCandidate& cand,
               const reco::Vertex* pv = 0 ) const {
    if ( ( ( mMin > 0 ) && ( mMax < 0 ) ) ||
         ( ( mMin < 0 ) && ( mMax > 0 ) ) ||
         ( ( mMin > 0 ) && ( mMax > 0 ) && ( mMin < mMax ) ) ) {
      float mass = cand.mass();
      if (   mass < mMin   ) return false;
      if ( ( mMax >    0 ) &&
           ( mass > mMax ) ) return false;
    }
    if (         cand.      pt()   < pMin   ) return false;
    if ( ( eMax                    >    0 ) &&
         ( fabs( cand.     eta() ) > eMax ) ) return false;
    if ( ( yMax                    >    0 ) &&
         ( fabs( cand.rapidity() ) > yMax ) ) return false;
    return true;
  }

 private:
  float mMin;
  float mMax;
  float pMin;
  float eMax;
  float yMax;
};


class BPHFittedBasicSelect: public BPHHistoSpecificDecay::CandidateSelect {
 public:
  BPHFittedBasicSelect( float     massMin,
                        float     massMax,
                        float       ptMin = -1.0,
                        float      etaMax = -1.0,
                        float rapidityMax = -1.0 ): mMin(     massMin ),
                                                    mMax(     massMax ),
                                                    pMin(       ptMin ),
                                                    eMax(      etaMax ),
                                                    rMax( rapidityMax ) {
  }
  bool accept( const pat::CompositeCandidate& cand,
               const reco::Vertex* pv = 0 ) const {
    if ( !cand.hasUserFloat( "fitMass"     ) ) return false;
    float mass = cand.userFloat( "fitMass" );
    if ( ( ( mMin > 0 ) && ( mMax < 0 ) ) ||
         ( ( mMin < 0 ) && ( mMax > 0 ) ) ||
         ( ( mMin > 0 ) && ( mMax > 0 ) && ( mMin < mMax ) ) ) {
      if (   mass < mMin   ) return false;
      if ( ( mMax >    0 ) &&
           ( mass > mMax ) ) return false;
    }
    const Vector3DBase<float,GlobalTag>* fmom = BPHUserData::get
        < Vector3DBase<float,GlobalTag> >( cand, "fitMomentum" );
    if ( fmom == 0 ) return false;
    if ( pMin > 0 ) {
      if ( fmom->transverse() < pMin ) return false;
    }
    if ( eMax > 0 ) {
      if ( fabs( fmom->eta() ) > eMax ) return false;
    }
    if ( rMax > 0 ) {
      float x = fmom->x();
      float y = fmom->y();
      float z = fmom->z();
      float e = sqrt( ( x * x ) + ( y * y ) + ( z * z ) + ( mass * mass ) );
      float r = log( ( e + z ) / ( e - z ) ) / 2;
      if ( fabs( r ) > rMax ) return false;
    }
    return true;
  }

 private:
  float mMin;
  float mMax;
  float pMin;
  float eMax;
  float rMax;
};


class BPHCompositeVertexSelect: public BPHHistoSpecificDecay::CandidateSelect {
 public:
  BPHCompositeVertexSelect( float probMin,
                            float  cosMin = -1.0,
                            float  sigMin = -1.0 ): pMin( probMin ),
                                                    cMin(  cosMin ),
                                                    sMin(  sigMin ) {
  }
  bool accept( const pat::CompositeCandidate& cand,
               const reco::Vertex* pvtx = 0 ) const {
    const reco::Vertex* svtx = BPHUserData::get
         <reco::Vertex>( cand, "vertex" );
    if ( svtx == 0 ) return false;
    if ( pvtx == 0 ) return false;
    if ( pMin > 0 ) {
      if ( ChiSquaredProbability( svtx->chi2(),
                                  svtx->ndof() ) < pMin ) return false;
    }
    if ( ( cMin > 0 ) || ( sMin > 0 ) ) {
      TVector3 disp( svtx->x() - pvtx->x(),
                     svtx->y() - pvtx->y(),
                     0 );
      TVector3 cmom( cand.px(), cand.py(), 0 );
      float cosAlpha = disp.Dot( cmom ) / ( disp.Perp() * cmom.Perp() );
      if ( cosAlpha < cMin ) return false;
      if ( sMin < 0 ) return true;
      float mass = cand.mass();
      AlgebraicVector3 vmom( cand.px(), cand.py(), 0 );
      VertexDistanceXY vdistXY;
      Measurement1D distXY = vdistXY.distance( *svtx, *pvtx );
      double ctauPV = distXY.value() * cosAlpha * mass / cmom.Perp();
      GlobalError sve = svtx->error();
      GlobalError pve = pvtx->error();
      AlgebraicSymMatrix33 vXYe = sve.matrix() + pve.matrix();
      double ctauErrPV = sqrt( ROOT::Math::Similarity( vmom, vXYe ) ) * mass /
                               cmom.Perp2();
      if ( ( ctauPV / ctauErrPV ) < sMin ) return false;
    }
    return true;
  }

 private:
  float pMin;
  float cMin;
  float sMin;
};


class BPHFittedVertexSelect: public BPHHistoSpecificDecay::CandidateSelect {
 public:
  BPHFittedVertexSelect( float probMin,
                         float  cosMin = -1.0,
                         float  sigMin = -1.0 ): pMin( probMin ),
                                                 cMin(  cosMin ),
                                                 sMin(  sigMin ) {
  }
  bool accept( const pat::CompositeCandidate& cand,
               const reco::Vertex* pvtx ) const {
    const reco::Vertex* svtx = BPHUserData::get
         <reco::Vertex>( cand, "fitVertex" );
    if ( svtx == 0 ) return false;
    if ( pMin > 0 ) {
      if ( ChiSquaredProbability( svtx->chi2(),
                                  svtx->ndof() ) < pMin ) return false;
    }
    if ( ( cMin > 0 ) || ( sMin > 0 ) ) {
      TVector3 disp( svtx->x() - pvtx->x(),
                     svtx->y() - pvtx->y(),
                     0 );
      const Vector3DBase<float,GlobalTag>* fmom = BPHUserData::get
          < Vector3DBase<float,GlobalTag> >( cand, "fitMomentum" );
      if ( fmom == 0 ) return false;
      TVector3 cmom( fmom->x(), fmom->y(), 0 );
      float cosAlpha = disp.Dot( cmom ) / ( disp.Perp() * cmom.Perp() );
      if ( cosAlpha < cMin ) return false;
      if ( sMin < 0 ) return true;
      if ( !cand.hasUserFloat( "fitMass" ) ) return false;
      float mass = cand.userFloat( "fitMass" );
      AlgebraicVector3 vmom( fmom->x(), fmom->y(), 0 );
      VertexDistanceXY vdistXY;
      Measurement1D distXY = vdistXY.distance( *svtx, *pvtx );
      double ctauPV = distXY.value() * cosAlpha * mass / cmom.Perp();
      GlobalError sve = svtx->error();
      GlobalError pve = pvtx->error();
      AlgebraicSymMatrix33 vXYe = sve.matrix() + pve.matrix();
      double ctauErrPV = sqrt( ROOT::Math::Similarity( vmom, vXYe ) ) * mass /
                               cmom.Perp2();
      if ( ( ctauPV / ctauErrPV ) < sMin ) return false;
    }
    return true;
  }

 private:
  float pMin;
  float cMin;
  float sMin;
};


BPHHistoSpecificDecay::BPHHistoSpecificDecay( const edm::ParameterSet& ps ) {

  useOnia = ( SET_LABEL( oniaCandsLabel, ps ) != "" );
  useSd   = ( SET_LABEL(   sdCandsLabel, ps ) != "" );
  useSs   = ( SET_LABEL(   ssCandsLabel, ps ) != "" );
  useBu   = ( SET_LABEL(   buCandsLabel, ps ) != "" );
  useBd   = ( SET_LABEL(   bdCandsLabel, ps ) != "" );
  useBs   = ( SET_LABEL(   bsCandsLabel, ps ) != "" );
  if ( useOnia ) consume< vector<pat::CompositeCandidate> >( oniaCandsToken,
                                                             oniaCandsLabel );
  if ( useSd   ) consume< vector<pat::CompositeCandidate> >(   sdCandsToken,
                                                               sdCandsLabel );
  if ( useSs   ) consume< vector<pat::CompositeCandidate> >(   ssCandsToken,
                                                               ssCandsLabel );
  if ( useBu   ) consume< vector<pat::CompositeCandidate> >(   buCandsToken,
                                                               buCandsLabel );
  if ( useBd   ) consume< vector<pat::CompositeCandidate> >(   bdCandsToken,
                                                               bdCandsLabel );
  if ( useBs   ) consume< vector<pat::CompositeCandidate> >(   bsCandsToken,
                                                               bsCandsLabel );

  static const BPHSoftMuonSelect sms;

  double  phiMassMin =  0.85;
  double  phiMassMax =  3.30;
  double  phiPtMin   = 16.0;
  double  phiEtaMax  = -1.0;
  double  phiRMax    = -1.0;
  double jPsiMassMin =  2.95;
  double jPsiMassMax =  3.30;
  double jPsiPtMin   = 16.0;
  double jPsiEtaMax  = -1.0;
  double jPsiRMax    = -1.0;
  double psi2MassMin =  3.40;
  double psi2MassMax =  4.00;
  double psi2PtMin   = 13.0;
  double psi2EtaMax  = -1.0;
  double psi2RMax    = -1.0;
  double upsMassMin  =  8.50;
  double upsMassMax  = 11.0;
  double upsPtMin    = 13.0;
  double upsEtaMax   = -1.0;
  double upsRMax     = -1.0;

  double oniaProbMin =  0.005;
  double oniaCosMin  = -1.0;
  double oniaSigMin  = -1.0;

  double oniaMuPtMinLoose  = -1.0;
  double oniaMuPtMinTight  = -1.0;
  double oniaMuEtaMaxLoose = -1.0;
  double oniaMuEtaMaxTight = -1.0;

   phiBasicSelect      = new BPHCompositeBasicSelect(
                              phiMassMin,  phiMassMax,
                              phiPtMin  ,  phiEtaMax ,  phiRMax );
  jPsiBasicSelect      = new BPHCompositeBasicSelect(
                             jPsiMassMin, jPsiMassMax,
                             jPsiPtMin  , jPsiEtaMax , jPsiRMax );
  psi2BasicSelect      = new BPHCompositeBasicSelect(
                             psi2MassMin, psi2MassMax,
                             psi2PtMin  , psi2EtaMax , psi2RMax );
  upsBasicSelect       = new BPHCompositeBasicSelect(
                              upsMassMin,  upsMassMax,
                              upsPtMin  ,  upsEtaMax , upsRMax );
  oniaVertexSelect     = new BPHCompositeVertexSelect(
                             oniaProbMin, oniaCosMin, oniaSigMin );
  oniaDaughterSelect   = new BPHDaughterSelect(
                             oniaMuPtMinLoose , oniaMuPtMinTight ,
                             oniaMuEtaMaxLoose, oniaMuEtaMaxTight, &sms );

  double buJPsiMassMin   = BPHParticleMasses::jPsiMass - 0.150;
  double buJPsiMassMax   = BPHParticleMasses::jPsiMass + 0.150;
  double buJPsiPtMin     =  8.0;
  double buJPsiEtaMax    = -1.0;
  double buJPsiRMax      = -1.0;
  double buProbMin       =  0.10;
  double buCosMin        =  0.99;
  double buSigMin        =  3.0;
  double buMuPtMinLoose  =  4.0;
  double buMuPtMinTight  =  4.0;
  double buMuEtaMaxLoose =  2.2;
  double buMuEtaMaxTight =  2.2;

  buKPtMin = 1.6;

  buJPsiBasicSelect    = new BPHCompositeBasicSelect(
                             buJPsiMassMin, buJPsiMassMax,
                             buJPsiPtMin  , buJPsiEtaMax , buJPsiRMax );
  buVertexSelect       = new BPHFittedVertexSelect(
                             buProbMin, buCosMin, buSigMin );
  buJPsiDaughterSelect = new BPHDaughterSelect(
                             buMuPtMinLoose , buMuPtMinTight ,
                             buMuEtaMaxLoose, buMuEtaMaxTight, &sms );

  double bdJPsiMassMin   = BPHParticleMasses::jPsiMass - 0.150;
  double bdJPsiMassMax   = BPHParticleMasses::jPsiMass + 0.150;
  double bdJPsiPtMin     =  8.0;
  double bdJPsiEtaMax    = -1.0;
  double bdJPsiRMax      = -1.0;
  double bdKx0MassMin    = BPHParticleMasses::kx0Mass - 0.075;
  double bdKx0MassMax    = BPHParticleMasses::kx0Mass + 0.075;
  double bdKx0PtMin      = -1.0;
  double bdKx0EtaMax     = -1.0;
  double bdKx0RMax       = -1.0;
  double bdProbMin       =  0.10;
  double bdCosMin        =  0.99;
  double bdSigMin        =  3.0;
  double bdMuPtMinLoose  =  4.0;
  double bdMuPtMinTight  =  4.0;
  double bdMuEtaMaxLoose =  2.2;
  double bdMuEtaMaxTight =  2.2;

  bdJPsiBasicSelect    = new BPHCompositeBasicSelect(
                             bdJPsiMassMin, bdJPsiMassMax,
                             bdJPsiPtMin  , bdJPsiEtaMax , bdJPsiRMax );
  bdKx0BasicSelect     = new BPHCompositeBasicSelect(
                             bdKx0MassMin, bdKx0MassMax,
                             bdKx0PtMin  , bdKx0EtaMax , bdKx0RMax );
  bdVertexSelect       = new BPHFittedVertexSelect(
                             bdProbMin, bdCosMin, bdSigMin );
  bdJPsiDaughterSelect = new BPHDaughterSelect(
                             bdMuPtMinLoose , bdMuPtMinTight ,
                             bdMuEtaMaxLoose, bdMuEtaMaxTight, &sms );

  double bsJPsiMassMin   = BPHParticleMasses::jPsiMass - 0.150;
  double bsJPsiMassMax   = BPHParticleMasses::jPsiMass + 0.150;
  double bsJPsiPtMin     =  8.0;
  double bsJPsiEtaMax    = -1.0;
  double bsJPsiRMax      = -1.0;
  double bsPhiMassMin    = BPHParticleMasses::phiMass - 0.010;
  double bsPhiMassMax    = BPHParticleMasses::phiMass + 0.010;
  double bsPhiPtMin      = -1.0;
  double bsPhiEtaMax     = -1.0;
  double bsPhiRMax       = -1.0;
  double bsProbMin       =  0.10;
  double bsCosMin        =  0.99;
  double bsSigMin        =  3.0;
  double bsMuPtMinLoose  =  4.0;
  double bsMuPtMinTight  =  4.0;
  double bsMuEtaMaxLoose =  2.2;
  double bsMuEtaMaxTight =  2.2;
  bsJPsiBasicSelect    = new BPHCompositeBasicSelect(
                             bsJPsiMassMin, bsJPsiMassMax,
                             bsJPsiPtMin  , bsJPsiEtaMax , bsJPsiRMax );
  bsPhiBasicSelect     = new BPHCompositeBasicSelect(
                             bsPhiMassMin, bsPhiMassMax,
                             bsPhiPtMin  , bsPhiEtaMax , bsPhiRMax );
  bsVertexSelect       = new BPHFittedVertexSelect(
                             bsProbMin, bsCosMin, bsSigMin );
  bsJPsiDaughterSelect = new BPHDaughterSelect(
                             bsMuPtMinLoose , bsMuPtMinTight ,
                             bsMuEtaMaxLoose, bsMuEtaMaxTight, &sms );

}


BPHHistoSpecificDecay::~BPHHistoSpecificDecay() {

  delete  phiBasicSelect;
  delete jPsiBasicSelect;
  delete psi2BasicSelect;
  delete  upsBasicSelect;
  delete oniaVertexSelect;
  delete oniaDaughterSelect;

  delete buJPsiBasicSelect;
  delete buVertexSelect;
  delete buJPsiDaughterSelect;

  delete bdJPsiBasicSelect;
  delete bdKx0BasicSelect;
  delete bdVertexSelect;
  delete bdJPsiDaughterSelect;

  delete bsJPsiBasicSelect;
  delete bsPhiBasicSelect;
  delete bsVertexSelect;
  delete bsJPsiDaughterSelect;

}


void BPHHistoSpecificDecay::fillDescriptions(
                            edm::ConfigurationDescriptions& descriptions ) {
   edm::ParameterSetDescription desc;
   desc.add<string>( "oniaCandsLabel", "" );
   desc.add<string>(   "sdCandsLabel", "" );
   desc.add<string>(   "ssCandsLabel", "" );
   desc.add<string>(   "buCandsLabel", "" );
   desc.add<string>(   "bdCandsLabel", "" );
   desc.add<string>(   "bsCandsLabel", "" );
   descriptions.add( "bphHistoSpecificDecay", desc );
   return;
}


void BPHHistoSpecificDecay::beginJob() {
  createHisto( "massPhi"    ,  35, 0.85, 1.20 ); // Phi  mass
  createHisto( "massJPsi"   ,  35, 2.95, 3.30 ); // JPsi mass
  createHisto( "massPsi2"   ,  60, 3.40, 4.00 ); // Psi2 mass
  createHisto( "massUps123" , 125, 8.50, 11.0 ); // Ups  mass
  createHisto( "massBu"     ,  50, 5.00, 6.00 ); // Bu   mass
  createHisto( "massBd"     ,  50, 5.00, 6.00 ); // Bd   mass
  createHisto( "massBs"     ,  50, 5.00, 6.00 ); // Bs   mass
  createHisto( "mfitBu"     ,  50, 5.00, 6.00 ); // Bu   mass, with constraint
  createHisto( "mfitBd"     ,  50, 5.00, 6.00 ); // Bd   mass, with constraint
  createHisto( "mfitBs"     ,  50, 5.00, 6.00 ); // Bs   mass, with constraint
  createHisto( "massBuJPsi" ,  35, 2.95, 3.30 ); // JPsi mass in Bu decay
  createHisto( "massBdJPsi" ,  35, 2.95, 3.30 ); // JPsi mass in Bd decay
  createHisto( "massBsJPsi" ,  35, 2.95, 3.30 ); // JPsi mass in Bs decay
  createHisto( "massBsPhi"  ,  50, 1.01, 1.03 ); // Phi  mass in Bs decay
  createHisto( "massBdKx0"  ,  50, 0.80, 1.05 ); // Kx0  mass in Bd decay

  createHisto( "massFull"   , 200, 2.00, 12.0 ); // Full onia mass

  return;
}

void BPHHistoSpecificDecay::analyze( const edm::Event& ev,
                                     const edm::EventSetup& es ) {

  // get magnetic field
  edm::ESHandle<MagneticField> magneticField;
  es.get<IdealMagneticFieldRecord>().get( magneticField );

  // get object collections
  // collections are got through "BPHTokenWrapper" interface to allow
  // uniform access in different CMSSW versions

  //////////// quarkonia ////////////

  edm::Handle< vector<pat::CompositeCandidate> > oniaCands;
  int iqo;
  int nqo = 0;
  if ( useOnia ) {
    oniaCandsToken.get( ev, oniaCands );
    nqo = oniaCands->size();
  }

  for ( iqo = 0; iqo < nqo; ++ iqo ) {
    LogTrace( "DataDump" )
           << "*********** quarkonium " << iqo << "/" << nqo << " ***********";
    const pat::CompositeCandidate& cand = oniaCands->at( iqo );
    if ( !oniaVertexSelect->accept( cand,
                                    BPHUserData::getByRef<reco::Vertex>( cand,
                                    "primaryVertex" ) ) ) continue;
    if ( !oniaDaughterSelect->accept( cand ) ) continue;
    fillHisto( "Full", cand );
    if (  phiBasicSelect->accept( cand ) ) fillHisto( "Phi"   , cand );
    if ( jPsiBasicSelect->accept( cand ) ) fillHisto( "JPsi"  , cand );
    if ( psi2BasicSelect->accept( cand ) ) fillHisto( "Psi2"  , cand );
    if (  upsBasicSelect->accept( cand ) ) fillHisto( "Ups123", cand );
  }

  //////////// Bu ////////////

  edm::Handle< vector<pat::CompositeCandidate> > buCands;
  int ibu;
  int nbu = 0;
  if ( useBu ) {
    buCandsToken.get( ev, buCands );
    nbu = buCands->size();
  }

  for ( ibu = 0; ibu < nbu; ++ ibu ) {
    LogTrace( "DataDump" )
           << "*********** Bu " << ibu << "/" << nbu << " ***********";
    const pat::CompositeCandidate& cand = buCands->at( ibu );
    const pat::CompositeCandidate* jPsi = BPHUserData::getByRef
         <pat::CompositeCandidate>( cand, "refToJPsi" );
    LogTrace( "DataDump" )
           << "JPsi: " << jPsi;
    if ( jPsi == 0 ) continue;
    if ( !buJPsiBasicSelect   ->accept( *jPsi ) ) continue;
    if ( !buJPsiDaughterSelect->accept( *jPsi ) ) continue;
    if ( !buVertexSelect->accept( cand,
                                  BPHUserData::getByRef<reco::Vertex>( *jPsi,
                                  "primaryVertex" ) ) ) continue;
    const reco::Candidate* kptr = BPHDaughters::get( cand, 0.49, 0.50 ).front();
    if ( kptr == 0 ) continue;
    if ( kptr->pt() < buKPtMin ) continue;
    fillHisto( "Bu"    ,  cand );
    fillHisto( "BuJPsi", *jPsi );
  }

  //////////// Bd ////////////

  edm::Handle< vector<pat::CompositeCandidate> > bdCands;
  int ibd;
  int nbd = 0;
  if ( useBd ) {
    bdCandsToken.get( ev, bdCands );
    nbd = bdCands->size();
  }

  for ( ibd = 0; ibd < nbd; ++ ibd ) {
    LogTrace( "DataDump" )
           << "*********** Bd " << ibd << "/" << nbd << " ***********";
    const pat::CompositeCandidate& cand = bdCands->at( ibd );
    const pat::CompositeCandidate* jPsi = BPHUserData::getByRef
         <pat::CompositeCandidate>( cand, "refToJPsi" );
    LogTrace( "DataDump" )
           << "JPsi: " << jPsi;
    if ( jPsi == 0 ) continue;
    const pat::CompositeCandidate* kx0 = BPHUserData::getByRef
         <pat::CompositeCandidate>( cand, "refToKx0" );
    LogTrace( "DataDump" )
           << "Kx0: " << kx0;
    if ( kx0 == 0 ) continue;
    if ( !bdJPsiBasicSelect   ->accept( *jPsi ) ) continue;
    if ( !bdKx0BasicSelect    ->accept( * kx0 ) ) continue;
    if ( !bdJPsiDaughterSelect->accept( *jPsi ) ) continue;
    if ( !bdVertexSelect->accept( cand,
                                  BPHUserData::getByRef<reco::Vertex>( *jPsi,
                                  "primaryVertex" ) ) ) continue;
    fillHisto( "Bd"    ,  cand );
    fillHisto( "BdJPsi", *jPsi );
    fillHisto( "BdKx0" , *kx0  );
  }

  //////////// Bs ////////////

  edm::Handle< vector<pat::CompositeCandidate> > bsCands;
  int ibs;
  int nbs = 0;
  if ( useBs ) {
    bsCandsToken.get( ev, bsCands );
    nbs = bsCands->size();
  }

  for ( ibs = 0; ibs < nbs; ++ ibs ) {
    LogTrace( "DataDump" )
           << "*********** Bs " << ibs << "/" << nbs << " ***********";
    const pat::CompositeCandidate& cand = bsCands->at( ibs );
    const pat::CompositeCandidate* jPsi = BPHUserData::getByRef
         <pat::CompositeCandidate>( cand, "refToJPsi" );
    LogTrace( "DataDump" )
           << "JPsi: " << jPsi;
    if ( jPsi == 0 ) continue;
    const pat::CompositeCandidate* phi = BPHUserData::getByRef
         <pat::CompositeCandidate>( cand, "refToPhi" );
    LogTrace( "DataDump" )
           << "Phi: " << phi;
    if ( phi == 0 ) continue;
    if ( !bsJPsiBasicSelect   ->accept( *jPsi ) ) continue;
    if ( !bsPhiBasicSelect    ->accept( * phi ) ) continue;
    if ( !bsJPsiDaughterSelect->accept( *jPsi ) ) continue;
    if ( !bsVertexSelect->accept( cand,
                                  BPHUserData::getByRef<reco::Vertex>( *jPsi,
                                  "primaryVertex" ) ) ) continue;
    fillHisto( "Bs"    ,  cand );
    fillHisto( "BsJPsi", *jPsi );
    fillHisto( "BsPhi" , *phi  );
  }


  return;

}


void BPHHistoSpecificDecay::endJob() {
  return;
}


void BPHHistoSpecificDecay::fillHisto( const string& name,
                                       const pat::CompositeCandidate& cand ) {
  float mass = ( cand.hasUserFloat( "fitMass" ) ?
                 cand.   userFloat( "fitMass" ) : -1 );
  fillHisto( "mass" + name, cand.mass() );
  fillHisto( "mfit" + name, mass );
  return;
}


void BPHHistoSpecificDecay::fillHisto( const string& name, float x ) {
  map<string,TH1F*>::iterator iter = histoMap.find( name );
  map<string,TH1F*>::iterator iend = histoMap.end();
  if ( iter == iend ) return;
  iter->second->Fill( x );
  return;
}


void BPHHistoSpecificDecay::createHisto( const string& name,
                                         int nbin, float hmin, float hmax ) {
  histoMap[name] = fs->make<TH1F>( name.c_str(), name.c_str(),
                                   nbin, hmin, hmax );
  return;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( BPHHistoSpecificDecay );
