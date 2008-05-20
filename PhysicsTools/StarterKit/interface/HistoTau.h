#ifndef StarterKit_HistoTau_h
#define StarterKit_HistoTau_h

//------------------------------------------------------------
// Title: HistoTau.h
// Purpose: To histogram Taus
//
// Authors:
// Liz Sexton-Kennedy <sexton@fnal.gov>
// Eric Vaandering <ewv@fnal.gov >
// Petar Maksimovic <petar@jhu.edu>
// Sal Rappoccio <rappocc@fnal.gov>
//------------------------------------------------------------
//
// Interface:
//
//   HistoTau ( TFile * file );
//   Description: Constructor.
//
//   void fill( TK::Tau * );
//   Description: Fill object. Will fill relevant muon variables
//
//   void write();
//   Description: Write object to file in question.
//
//   ~HistoTau
//    Description: Destructor. Deallocates memory.
//
//------------------------------------------------------------
//
// Modification History:
//
//   -29Nov07: Sal Rappoccio: Creation of the object
//------------------------------------------------------------

// This package's include files
#include "PhysicsTools/StarterKit/interface/HistoGroup.h"
#include "PhysicsTools/StarterKit/interface/HistoTrack.h"

// CMSSW include files
#include "DataFormats/PatCandidates/interface/Tau.h"


// STL include files
#include <string>
#include <vector>

// ROOT include files
#include <TH1D.h>

namespace pat {

  class HistoTau : public HistoGroup<Tau> {

  public:
    HistoTau(std::string dir = "tau", std::string group = "Tau",std::string pre="tau",
		   double pt1=0, double pt2=200, double m1=0, double m2=200 );
    virtual ~HistoTau() { } ;


    // fill a plain ol' tau:
    virtual void fill( const Tau *tau, uint iPart = 1, double weight = 1.0 );
    virtual void fill( const Tau &tau, uint iPart = 1, double weight = 1.0 ) { fill(&tau, iPart, weight); }

    // fill a tau that is a shallow clone, and take kinematics from 
    // shallow clone but detector plots from the tau itself
    virtual void fill( const reco::ShallowCloneCandidate *tau, uint iPart = 1, double weight = 1.0 );
    virtual void fill( const reco::ShallowCloneCandidate &tau, uint iPart = 1, double weight = 1.0 )
    { fill(&tau, iPart, weight); }

    virtual void fillCollection( const std::vector<Tau> & coll, double weight = 1.0 );

    // Clear ntuple cache
    void clearVec();

  protected:
    HistoTrack   * histoLeadingTrack_;        //!< Leading track
    HistoTrack   * histoSignalTrack_;         //!< Tracks in signal cone
    HistoTrack   * histoIsolationTrack_;      //!< Tracks in isolation cone
    PhysVarHisto * h_emEnergyFraction_ ;      //!< EM Fraction
    PhysVarHisto * h_eOverP_  ;               //!< E over P
  };
}
#endif
