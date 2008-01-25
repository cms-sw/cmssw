#ifndef StarterKit_HistoMuon_h
#define StarterKit_HistoMuon_h

//------------------------------------------------------------
// Title: HistoMuon.h
// Purpose: To histogram Muons
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
//   HistoMuon ( TFile * file );
//   Description: Constructor.
//
//   void fill( TK::Muon * );
//   Description: Fill object. Will fill relevant muon variables
//
//   void write();
//   Description: Write object to file in question.
//
//   ~HistoMuon
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

// CMSSW include files
#include "DataFormats/PatCandidates/interface/Muon.h"


// STL include files
#include <string>
#include <vector>

// ROOT include files
#include <TH1D.h>

namespace pat {

  class HistoMuon : public HistoGroup<Muon> {

  public:
    HistoMuon(std::string dir = "muon",
		   double pt1=0, double pt2=200, double m1=0, double m2=200 );
    virtual ~HistoMuon() { } ;

    void fill( const Muon *muon, uint iPart = 0 );

    // &&& Isn't this one already provided in the base class?
    void fill( const Muon &muon, uint iPart = 0 ) { fill(&muon, iPart); }

    // Clear ntuple cache
    void clearVec();

  protected:
    PhysVarHisto * h_trackIso_ ;   //!<   &&& document this
    PhysVarHisto * h_caloIso_  ;   //!<   &&& document this
    PhysVarHisto * h_leptonID_ ;   //!<   &&& document this
  };
}
#endif
