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
//!\brief CompositeKit is a plotting kit to plot a collection of CompositeCandidates
//!
//!
//-------------------------------------------------------------------------------------
//
// Original Author:  Sal Rappoccio
//         Created:  Wed Nov 28 15:31:57 CST 2007
// $Id: CompositeKit.h,v 1.4 2008/07/08 21:53:11 srappocc Exp $
//
// Revision History:
//       -  Sal Rappoccio, Fri Nov 30 12:49:44 CST 2007: Added other objects as first
//          stab of full analysis toolkit.
//


// system include files
#include <memory>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/EDProducer.h"
#include "PhysicsTools/StarterKit/interface/PatKitHelper.h"
#include "PhysicsTools/StarterKit/interface/HistoComposite.h"
#include "PhysicsTools/StarterKit/interface/PhysVarHisto.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"

//
// class declaration
//

class CompositeKit : public edm::EDProducer
{
public:
  explicit CompositeKit(const edm::ParameterSet&);
  virtual ~CompositeKit();

protected:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void produce( edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  // ----------member data ---------------------------

  bool                       verboseLevel_;

  // Input tag
  edm::InputTag              src_;

  // description
  std::string                description_;

  // The main sub-object which does the real work
  pat::PatKitHelper          helper_;



  // Physics objects handles
  edm::Handle<std::vector<pat::Muon> >     muonHandle_;
  edm::Handle<std::vector<pat::Electron> > electronHandle_;
  edm::Handle<std::vector<pat::Tau> >      tauHandle_;
  edm::Handle<std::vector<pat::Jet> >      jetHandle_;
  edm::Handle<std::vector<pat::MET> >      METHandle_;
  edm::Handle<std::vector<pat::Photon> >   photonHandle_;

  // Composite Candidates Input handle
  edm::Handle<std::vector<reco::CompositeCandidate> >  compositeCandHandle_;

  // Composite variable histograms
  // NOTE! Should be replaced in the future with configurable settings
  pat::HistoComposite *      compositeCandHist_;
  
  // List of variables to ntuplize. This will also
  // go away when we have ExpressionHistograms
  std::vector<pat::PhysVarHisto *>  compositeNtVars_; 
 
};


#endif
