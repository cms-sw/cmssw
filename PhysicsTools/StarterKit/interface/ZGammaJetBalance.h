#ifndef ZGammaJetBalanceKit_h
#define ZGammaJetBalanceKit_h

// -*- C++ -*-
//
// Package:    ZGammaJetBalanceKit
// Class:      ZGammaJetBalanceKit
//
/**


*/
//-------------------------------------------------------------------------------------
//!\class ZGammaJetBalanceKit ZGammaJetBalanceKit.cc Demo/TempAnaToolkit/test/ZGammaJetBalanceKit.cc
//!\brief ZGammaJetBalanceKit is an ED analyzer with histograms appropriate for lepton+jet+MET.
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
// $Id: ZGammaJetBalance.h,v 1.1 2008/05/14 05:51:54 srappocc Exp $
//
// Revision History:
//       -  Malina Kirn, v0.9, Wed Jan 23 12:31:57 EST 2008:
//          Modified HZZKitDemo to produce composite objects from semi-leptonic data.
//


// system include files
#include <memory>
#include <fstream>

// user include files
#include "PhysicsTools/StarterKit/interface/StarterKit.h"
#include "PhysicsTools/StarterKit/interface/HistoComposite.h"

//
// class declaration
//

class ZGammaJetBalanceKit : public StarterKit 
{
public:
  explicit ZGammaJetBalanceKit(const edm::ParameterSet&);
  virtual ~ZGammaJetBalanceKit();
    
protected:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void produce( edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  // ----------member data ---------------------------

  edm::Handle<std::vector<reco::NamedCompositeCandidate> > zHandle_;
  edm::InputTag                                            zHandleName_;

  // Event histograms
  pat::PhysVarHisto  *  ptJet1OverPtZ_;
  pat::PhysVarHisto  *  ptJet1OverPtGamma_;
  pat::HistoComposite *   zHistos_;

  
  // List of variables to ntuplize. This will also
  // go away when we have ExpressionHistograms
  std::vector<pat::PhysVarHisto *>  zgammaNtVars_;  
  
};


#endif
