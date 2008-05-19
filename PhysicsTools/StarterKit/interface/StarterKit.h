#ifndef StarterKit_test_StarterKit_h
#define StarterKit_test_StarterKit_h



// -*- C++ -*-
//// -*- C++ -*-
//
// Package:    StarterKit
// Class:      StarterKit
//
/**


*/
//-------------------------------------------------------------------------------------
//!\class StarterKit StarterKit.cc PhysicsTools/StarterKit/src/StarterKit.cc
//!\brief StarterKit is a generic ED analyzer with predefined physics histograms.
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
// $Id: StarterKit.h,v 1.4 2008/04/14 18:56:05 srappocc Exp $
//
// Revision History:
//       -  Sal Rappoccio, Fri Nov 30 12:49:44 CST 2007: Added other objects as first
//          stab of full analysis toolkit.
//       -  Sal Rappoccio, Mon Mar 03 12:00:00 CST 2008: Added photons and taus
//       -  Sal Rappoccio, Mon Apr 14 13:45:59 CDT 2008: Added CSA07 information.


// system include files
#include <memory>
#include <fstream>

// user include files
#include "PhysicsTools/StarterKit/interface/PhysicsHistograms.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
//#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "PhysicsTools/UtilAlgos/interface/TFileDirectory.h"

//
// class declaration
//

class StarterKit : public edm::EDProducer  // public edm::EDAnalyzer
{
public:
  explicit StarterKit(const edm::ParameterSet&);
  virtual ~StarterKit();


  // Function to print out candidates
  friend std::ostream & operator<<( std::ostream & out, const reco::Candidate & cand );


protected:
  // Function to add ntuple variables to the EDProducer
  virtual void addNtupleVar ( std::string name, std::string type );

  // Save ntuple variables to event evt
  void saveNtuple ( const std::vector<pat::PhysVarHisto*> & ntvars,
		    edm::Event & evt );

  // Helper function template to write objects to event
  template <class T>
  void saveNtupleVar( std::string name, T value,
		      edm::Event & evt );

  // Helper function template to write vectors of objects to event
  template <class T>
  void saveNtupleVec( std::string name, const std::vector<T> & invec,
		      edm::Event & evt);

  // beginJob
  virtual void beginJob(const edm::EventSetup&) ;
  // produce is where the ntuples are made
  virtual void produce( edm::Event &, const edm::EventSetup & );
  // endJob
  virtual void endJob() ;

  // The main sub-object which does the real work
  PhysicsHistograms  physHistos_ ;

  // Output file (move elsewhere when its function is clear)
  std::string     outputTextName_ ;
  ofstream        outputFile_;
  // Verbosity
  int             verboseLevel_;
  // If isCSA07Soup is true, then the StarterKit will bring out some
  // event weighting information and information regarding the
  // sample composition
  bool            isCSA07Soup_;

  // Histogram server
  edm::Service<TFileService> fs;

  // Variables to ntuplize
  std::vector< pat::PhysVarHisto* > ntVars_ ;

  // Physics objects handles
  edm::Handle<std::vector<pat::Muon> >     muonHandle_;
  edm::Handle<std::vector<pat::Electron> > electronHandle_;
  edm::Handle<std::vector<pat::Tau> >      tauHandle_;
  edm::Handle<std::vector<pat::Jet> >      jetHandle_;
  edm::Handle<std::vector<pat::MET> >      METHandle_;
  edm::Handle<std::vector<pat::Photon> >   photonHandle_;

  // CSA07 "soup" specific information
  pat::PhysVarHisto *  h_csa07Id_;
  pat::PhysVarHisto *  h_csa07Weight_;

  // ----------member data ---------------------------
};



#endif
