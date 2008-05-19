#ifndef StarterKit_test_CompositeKit_h
#define StarterKit_test_CompositeKit_h

// -*- C++ -*-
//
// Package:    CompositeKit
// Class:      CompositeKit
//
/**


*/
//-------------------------------------------------------------------------------------
//!\class CompositeKit CompositeKit.cc PhysicsTools/StarterKit/test/CompositeKit.cc
//!\brief CompositeKit is a plotting kit to plot a collection of NamedCompositeCandidates
//!
//!
//-------------------------------------------------------------------------------------
//
// Original Author:  Sal Rappoccio
//         Created:  Wed Nov 28 15:31:57 CST 2007
// $Id: CompositeKit.h,v 1.1 2008/05/13 21:43:45 srappocc Exp $
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
#include "PhysicsTools/StarterKit/interface/PhysVarHisto.h"
#include "DataFormats/Candidate/interface/NamedCompositeCandidate.h"

//
// class declaration
//

class CompositeKit : public StarterKit
{
public:
  explicit CompositeKit(const edm::ParameterSet&);
  virtual ~CompositeKit();

protected:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void produce( edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  // ----------member data ---------------------------

  // Input tag
  edm::InputTag              compositeCandTag_;

  // description
  std::string                description_;

  // Input mass and pt ranges
  double pt1_, pt2_, m1_, m2_, mm1_, mm2_;

  // Input handle
  edm::Handle<std::vector<reco::NamedCompositeCandidate> >  compositeCandHandle_;

  // NamedComposite variable histograms
  // NOTE! Should be replaced in the future with configurable settings
  pat::HistoComposite *      compositeCandHist_;

  // temporary kludge to get the resonance mass
  // When we have ExpressionHistograms, this will go away
  pat::PhysVarHisto          * compositeCandMass_;
  std::string                  compositeCandMassName_;
  
  // List of variables to ntuplize. This will also
  // go away when we have ExpressionHistograms
  std::vector<pat::PhysVarHisto *>  compositeNtVars_;  
};


#endif
