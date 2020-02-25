// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TkJetParticle
// 

#include "DataFormats/L1TrackTrigger/interface/L1TkJetParticle.h"


using namespace l1t ;


L1TkJetParticle::L1TkJetParticle()
{
}

L1TkJetParticle::L1TkJetParticle( const LorentzVector& p4,
				  const edm::Ref< JetBxCollection >& jetRef,
				  const std::vector< edm::Ptr< L1TTTrackType > >& trkPtrs,
				  float jetvtx )
: L1Candidate( p4 ),
  jetRef_ ( jetRef ),
  trkPtrs_ ( trkPtrs ),
  JetVtx_ ( jetvtx )
{
}
L1TkJetParticle::L1TkJetParticle( const LorentzVector& p4,
//				  const edm::Ref< JetBxCollection >& jetRef,
				  const std::vector< edm::Ptr< L1TTTrackType > >& trkPtrs,
				  float jetvtx,
				  unsigned int ntracks,unsigned int tighttracks,unsigned int displacedtracks,unsigned int tightdisplacedtracks
				 // L1TkJetParticleDisp counters
				)
: L1Candidate( p4 ),
  //jetRef_ ( jetRef ),
  trkPtrs_ ( trkPtrs ),
  JetVtx_ ( jetvtx ),
  ntracks_(ntracks),
  tighttracks_(tighttracks),
  displacedtracks_(displacedtracks),
  tightdisplacedtracks_(tightdisplacedtracks)
{
                //L1TkJetParticleDisp DispCounters(ntracks,tighttracks, displacedtracks, tightdisplacedtracks);
		//setDispCounters(DispCounters);
}
int L1TkJetParticle::bx() const {

  // in the producer L1TkJetProducer.cc, we keep only jets with bx = 0 
  int dummy = 0;
  return dummy;
  
  /*
    int dummy = -999;
    if ( jetRef_.isNonnull() ) {
    return (getJetRef() -> bx()) ;
    }
    else {
    return dummy;
    
    }
  */
}

