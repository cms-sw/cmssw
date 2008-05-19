#ifndef TtSemiEvtKit_h
#define TtSemiEvtKit_h

// -*- C++ -*-
//
// Package:    TtSemiEvtKit
// Class:      TtSemiEvtKit
//
/**


*/
//-------------------------------------------------------------------------------------
//!\class TtSemiEvtKit TtSemiEvtKit.cc Demo/TempAnaToolkit/test/TtSemiEvtKit.cc
//!\brief TtSemiEvtKit is an ED analyzer with histograms appropriate for lepton+jet+MET.
//!
//!  This is an ED analyzer which creates and fills a bunch of
//!  histograms of various physics quantities.  However, in order to
//!  make this also work in FWLite, most of the actual work is
//!  performed by another object called PhysicsHistograms, to which this
//!  ED analyzer delegates most of the work, except the interactions with
//!  EDM and Framework, like:
//!
//!  - obtaining parameters from edm::ParameterSet.  PhysicsHistograms receives
//!    commands with lists of sub-components to manipulate (usually only enable or
//!    disable).
//!
//!  - fetching collections from the event (the iteration over collections is done
//!    by PhysicsHistograms).
//!
//!  - putting single numbers back to the event -- this is how flat ntuples
//!    are made. (PhysicsHistograms provides several lists of PhysVarHisto* pointers:
//!       1. all PhysVarHistos
//!       2. those that need to be histogrammed
//!       3. those that need to be ntupled
//!       4. those that need to be filled in each events (i.e. "active" ones, which
//!          is a union of (2) and (3), and a subset of (1).
//! 
//-------------------------------------------------------------------------------------
//
// Original Author:  Malina Kirn
//         Created:  Wed Jan 23 12:31:57 EST 2008
// $Id: TtSemiEvtKit.h,v 1.2 2008/05/10 06:55:46 srappocc Exp $
//
// Revision History:
//       -  Malina Kirn, v0.9, Wed Jan 23 12:31:57 EST 2008:
//          Modified HZZKitDemo to produce composite objects from semi-leptonic data.
//


// system include files
#include <memory>
#include <fstream>

// user include files
#include "PhysicsTools/StarterKit/interface/LepJetMetKit.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiEvtSolution.h"

//
// class declaration
//

class TtSemiEvtKit : public LepJetMetKit 
{
public:
  explicit TtSemiEvtKit(const edm::ParameterSet&);
  virtual ~TtSemiEvtKit();
    
protected:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void produce( edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  // ----------member data ---------------------------
  edm::InputTag evtsols;

  pat::HistoComposite        * histoTtSemiEvt_;  
  pat::HistoJet              * histoHadb_;
  pat::HistoJet              * histoHadq_;
  pat::HistoJet              * histoHadp_;
  pat::HistoJet              * histoLepb_;

  pat::PhysVarHisto          * histoLRJetCombProb_;
  pat::PhysVarHisto          * histoLRSignalEvtProb_;
  pat::PhysVarHisto          * histoKinFitProbChi2_;
  pat::PhysVarHisto          * histoTtMass_;


  std::vector<pat::PhysVarHisto *>  ttNtVars_;
};


#endif
