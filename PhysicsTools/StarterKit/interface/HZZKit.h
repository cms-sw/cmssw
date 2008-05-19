#ifndef StarterKit_test_HZZKit_h
#define StarterKit_test_HZZKit_h

// -*- C++ -*-
//
// Package:    HZZKit
// Class:      HZZKit
//
/**


*/
//-------------------------------------------------------------------------------------
//!\class HZZKit HZZKit.cc Demo/TempAnaToolkit/src/HZZKit.cc
//!\brief HZZKit is a generic ED analyzer with predefined physics histograms.
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
// Original Author:  Eric Vaandering
//         Created:  Wed Nov 28 15:31:57 CST 2007
// $Id: HZZKit.h,v 1.3 2008/04/11 16:31:17 srappocc Exp $
//
// Revision History:
//       -  Sal Rappoccio, Fri Nov 30 12:49:44 CST 2007: Added other objects as first
//          stab of full analysis toolkit.
//


// system include files
#include <memory>
#include <fstream>

// user include files
#include "PhysicsTools/StarterKit/interface/StarterKit.h"
#include "PhysicsTools/StarterKit/interface/HistoComposite.h"
#include "DataFormats/Candidate/interface/NamedCompositeCandidate.h"

//
// class declaration
//

class HZZKit : public StarterKit
{
public:
  explicit HZZKit(const edm::ParameterSet&);
  virtual ~HZZKit();

protected:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void produce( edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  // ----------member data ---------------------------

  edm::Handle<std::vector<reco::NamedCompositeCandidate> >  zToMuMuHandle_;
  edm::Handle<std::vector<reco::NamedCompositeCandidate> >  hToZZHandle_;

  // NamedComposite variable histograms
  // NOTE! Should be replaced in the future with configurable settings
  pat::HistoComposite *      zmumuCandHist_;    // Single Z->mumu
  pat::HistoComposite *      zeeCandHist_;      // Single Z->ee
  pat::HistoComposite *      hCandHist_;        // H->ZZ
};


#endif
