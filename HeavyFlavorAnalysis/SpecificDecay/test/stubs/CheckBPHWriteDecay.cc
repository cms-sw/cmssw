#include "HeavyFlavorAnalysis/SpecificDecay/test/stubs/CheckBPHWriteDecay.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoSelect.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHMomentumSelect.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHVertexSelect.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHTrackReference.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/PatCandidates/interface/GenericParticle.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicState.h"

#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

#include <TH1.h>
#include <TFile.h>

#include <set>
#include <string>
#include <iostream>
#include <fstream>

using namespace std;


CheckBPHWriteDecay::CheckBPHWriteDecay( const edm::ParameterSet& ps ) {

  if ( ps.exists( "runNumber" ) )
                   runNumber = ps.getParameter<unsigned int>( "runNumber" );
  else             runNumber = 0;
  if ( ps.exists( "evtNumber" ) )
                   evtNumber = ps.getParameter<unsigned int>( "evtNumber" );
  else             evtNumber = 0;
  candsLabel = ps.getParameter< std::vector<std::string> >( "candsLabel" );
  int i;
  int n =
  candsLabel.  size();
  candsToken.resize( n );
  for ( i = 0; i < n; ++i )
        consume< vector<pat::CompositeCandidate> >( candsToken[i],
                                                    candsLabel[i] );

  if ( ps.exists( "fileName" ) )
       osPtr = new ofstream( ps.getParameter<string>( "fileName" ) );
  else osPtr = &cout;


}


CheckBPHWriteDecay::~CheckBPHWriteDecay() {
}


void CheckBPHWriteDecay::beginJob() {
  return;
}


void CheckBPHWriteDecay::analyze( const edm::Event& ev,
                                  const edm::EventSetup& es ) {

  ostream& os = *osPtr;

  if ( ( runNumber != 0 ) && ( ev.id().run  () != runNumber ) ) return;
  if ( ( evtNumber != 0 ) && ( ev.id().event() != evtNumber ) ) return;
  os << "--------- event "
     << ev.id().run() << " / "
     << ev.id().event()
     << " ---------" << endl;

  int il;
  int nl =
  candsLabel.size();
  for ( il = 0; il < nl; ++il ) {
    edm::Handle< vector<pat::CompositeCandidate> > cands;
    candsToken[il].get( ev, cands );
    int ic;
    int nc = cands->size();
    for ( ic = 0; ic < nc; ++ ic ) {
      os << "*********** " << candsLabel[il] << " " << ic << "/" << nc
         << " ***********"
         << endl;
      const pat::CompositeCandidate& cand = cands->at( ic );
      dump( os, cand );
    }
  }
  return;

}


void CheckBPHWriteDecay::endJob() {
  return;
}


void CheckBPHWriteDecay::dump( std::ostream& os,
                               const pat::CompositeCandidate& cand ) {

  float mfit = ( cand.hasUserFloat( "fitMass" ) ?
                 cand.   userFloat( "fitMass" ) : -1 );
  os << &cand
     << " mass : " << cand.mass() << " " << mfit << " "
     << (   cand.hasUserData      ( "cowboy" ) ?
          ( cand.   userData<bool>( "cowboy" ) ? "cowboy" : "sailor" )
                                                   : "" ) << endl;
  writeMomentum( os, "cmom ", cand, false );
  writePosition( os, " xyz ", cand.momentum() );
  const reco::Vertex* vptr = 
        ( cand.hasUserData              ( "vertex" ) ? 
          cand.   userData<reco::Vertex>( "vertex" ) : 0 );
  if ( vptr != 0 ) {
    writePosition( os, "vpos : ", *vptr, false );
    os << " --- " << vptr->chi2() << " / " << vptr->ndof()
       << " ( " << ChiSquaredProbability( vptr->chi2(),
                                            vptr->ndof() ) << " ) " << endl;
  }
  const reco::Vertex* vfit = 
        ( cand.hasUserData              ( "fitVertex" ) ? 
          cand.   userData<reco::Vertex>( "fitVertex" ) : 0 );
  if ( vfit != 0 ) {
    writePosition( os, "vfit : ", *vfit, false );
    os << " --- "  << vfit->chi2() << " / " << vfit->ndof()
       << " ( " << ChiSquaredProbability( vfit->chi2(),
                                            vfit->ndof() ) << " ) " << endl;
  }
  if ( cand.hasUserData( "fitMomentum" ) )
       writePosition( os, "fmom : ",
      *cand.   userData< Vector3DBase<float,GlobalTag> >( "fitMomentum" ) );

  if ( cand.hasUserData( "primaryVertex" ) ) {
    const vertex_ref* pvr = cand.userData<vertex_ref>( "primaryVertex" );
    if ( pvr->isNonnull() ) {
      const reco::Vertex* pvtx = pvr->get();
      if ( pvtx != 0 ) writePosition( os, "ppos ", *pvtx );
    }
  }
  int i;
  int n = cand.numberOfDaughters();
  for ( i = 0; i < n; ++i ) {
    const reco::Candidate* dptr = cand.daughter( i );
    os << "daug " << i << "/" << n << " : " << dptr;
    writeMomentum( os, " == ", *dptr, false );
    os << " " << dptr->mass() << endl;
    const pat::Muon* mptr = dynamic_cast<const pat::Muon*>( dptr );
    os << "muon " << i << "/" << n << " : " << mptr << endl;
    const reco::Track* tptr = BPHTrackReference::getTrack( *dptr, "cfhpmnigs" );
    os << "trk  " << i << "/" << n << " : " << tptr;
    if ( tptr != 0 ) writeMomentum( os, " == ", *tptr );
    else             os << "no track" << endl;
  }
  const vector<string>& names = cand.userDataNames();
  int j;
  int m = names.size();
  for ( j = 0; j < m; ++j ) {
    const string& dname = names[j];
    if ( dname.substr( 0, 5 ) != "refTo" ) continue;
    const compcc_ref* ref = cand.userData<compcc_ref>( dname );
    os << dname << " : " << ref->get() << endl;
  }

  return;

}


#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( CheckBPHWriteDecay );
