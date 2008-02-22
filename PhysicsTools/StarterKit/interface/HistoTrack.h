#ifndef StarterKit_HistoTrack_h
#define StarterKit_HistoTrack_h

//------------------------------------------------------------
// Title: HistoTrack.h
// Purpose: To histogram Tracks
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
//   HistoTrack ( TFile * file );
//   Description: Constructor.
//
//   void fill( TK::Track * );
//   Description: Fill object. Will fill relevant track variables
//
//   void write();
//   Description: Write object to file in question.
//
//   ~HistoTrack
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
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"


// STL include files
#include <string>
#include <vector>

// ROOT include files
#include <TH1D.h>

namespace pat {

  class HistoTrack : public HistoGroup<reco::RecoChargedCandidate> {

  public:
    HistoTrack(std::string dir = "track", std::string groupName = "Track", std::string groupLabel = "track",
	       double pt1=0, double pt2=200, double m1=0, double m2=200 );
    virtual ~HistoTrack() { } ;

    void fill( const reco::RecoChargedCandidate *track, uint iPart = 0 );

    // &&& Isn't this one already provided in the base class?
    void fill( const reco::RecoChargedCandidate &track, uint iPart = 0 ) { fill(&track, iPart); }

    // Clear ntuple cache
    void clearVec();

  protected:
    PhysVarHisto * h_dxy_  ;    //!<   &&& document this
    PhysVarHisto * h_dz_   ;    //!<   &&& document this
  };
}
#endif
