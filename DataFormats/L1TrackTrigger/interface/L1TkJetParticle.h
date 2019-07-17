#ifndef L1TkTrigger_L1JetParticle_h
#define L1TkTrigger_L1JetParticle_h

// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TkJetParticle
// 

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/L1Trigger/interface/Jet.h"
//#include "DataFormats/L1TrackTrigger/interface/L1TkJetParticleDisp.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"



namespace l1t {
  
  class L1TkJetParticle : public L1Candidate
    {
      
    public:
      
      typedef TTTrack< Ref_Phase2TrackerDigi_ >  L1TTTrackType;
      typedef std::vector< L1TTTrackType > L1TTTrackCollection;
      
      L1TkJetParticle();
      
      L1TkJetParticle( const LorentzVector& p4,
		       const edm::Ref< JetBxCollection >& jetRef,
		       const std::vector< edm::Ptr< L1TTTrackType > >& trkPtrs,
		       float jetvtx = -999. );
      L1TkJetParticle( const LorentzVector& p4,
		       const std::vector< edm::Ptr< L1TTTrackType > >& trkPtrs,
		       float jetvtx = -999. ,
		unsigned int ntracks=0,unsigned int tighttracks=0,unsigned int displacedtracks=0,unsigned int tightdisplacedtracks=0	
			);
      
      virtual ~L1TkJetParticle() {}
      
      // ---------- const member functions ---------------------
      
      const edm::Ref< JetBxCollection >& getJetRef() const { return jetRef_ ; }
      
      const std::vector< edm::Ptr< L1TTTrackType > >& getTrkPtrs() const { return trkPtrs_; }
      
      float getJetVtx() const  { return JetVtx_ ; }
      unsigned int getNtracks() const {return ntracks_;}
      unsigned int getNTighttracks() const {return tighttracks_;}
      unsigned int getNDisptracks() const {return displacedtracks_;}
      unsigned int getNTightDisptracks() const {return tightdisplacedtracks_;} 
   
     // ---------- member functions ---------------------------
 //     void setDispCounters(L1TkJetParticleDisp counters){ counters_=counters;};
   //   L1TkJetParticleDisp getDispCounters() const { return counters_;};
      void setJetVtx(float JetVtx) { JetVtx_ = JetVtx ; }
            
      int bx() const;
      
    private:
      edm::Ref< JetBxCollection > jetRef_ ;
      std::vector< edm::Ptr< L1TTTrackType > >  trkPtrs_;
      //L1TkJetParticleDisp counters_;
      float JetVtx_ ;
     unsigned int ntracks_,tighttracks_,displacedtracks_,tightdisplacedtracks_;
    };
}

#endif

