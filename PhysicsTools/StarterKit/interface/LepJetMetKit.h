#ifndef LepJetMetKit_h
#define LepJetMetKit_h

// -*- C++ -*-
//
// Package:    LepJetMetKit
// Class:      LepJetMetKit
//
/**


*/
//-------------------------------------------------------------------------------------
//!\class LepJetMetKit LepJetMetKit.cc Demo/TempAnaToolkit/test/LepJetMetKit.cc
//!\brief LepJetMetKit is an ED analyzer with histograms appropriate for lepton+jet+MET.
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
// $Id: LepJetMetKit.h,v 1.1 2008/01/29 14:51:55 srappocc Exp $
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

class LepJetMetKit : public StarterKit 
{
public:
  explicit LepJetMetKit(const edm::ParameterSet&);
  virtual ~LepJetMetKit();
    
protected:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void produce( edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  // ----------member data ---------------------------
  
  bool lepJetMetEvent_;  // true if event contains at least one 'valid' lepton+jet+MET

  // Event histograms
  // NOTE! Should be replaced in the future with configurable settings
  pat::HistoComposite * leading2JetsHist_;    // 2 leading jets
  pat::HistoComposite * leading3JetsHist_;    // 3 leading jets
  pat::HistoComposite * leading4JetsHist_;    // 4 leading jets

  // Histogram parameters
  // # of bins on x axis
  unsigned int hNbinsJetMult_;    // jet multiplicity
  unsigned int hNbinsMuonMult_;   // muon multiplicity
  unsigned int hNbinsElecMult_;   // electron multiplicity
  unsigned int hNbinsLead2_;      // 2 leading jets
  unsigned int hNbinsLead3_;      // 3 leading jets
  unsigned int hNbinsLead4_;      // 4 leading jets

  // x axis minimum
  double hMinJetMult_;   // jet multiplicity (x-axis = #)
  double hMinMuonMult_;  // muon multiplicity (x-axis = #)
  double hMinElecMult_;  // electron multiplicity (x-axis = #)
  double hMinLead2_;     // 2 leading jets (x-axis = pT or mass)
  double hMinLead3_;     // 3 leading jets (x-axis = pT or mass)
  double hMinLead4_;     // 4 leading jets (x-axis = pT or mass)

  // x axis maximum
  double hMaxJetMult_;   // jet multiplicity (x-axis = #)
  double hMaxMuonMult_;  // muon multiplicity (x-axis = #)
  double hMaxElecMult_;  // electron multiplicity (x-axis = #)
  double hMaxLead2_;     // 2 leading jets (x-axis = pT or mass)
  double hMaxLead3_;     // 3 leading jets (x-axis = pT or mass)
  double hMaxLead4_;     // 4 leading jets (x-axis = pT or mass)
};


#endif
