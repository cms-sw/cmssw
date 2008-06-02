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


    // fill a plain ol' track:
    virtual void fill( const reco::RecoChargedCandidate *track, uint iPart = 1 );
    virtual void fill( const reco::RecoChargedCandidate &track, uint iPart = 1 ) { fill(&track, iPart); }

    // fill a track that is a shallow clone, and take kinematics from 
    // shallow clone but detector plots from the track itself
    virtual void fill( const reco::ShallowCloneCandidate *track, uint iPart = 1 );
    virtual void fill( const reco::ShallowCloneCandidate &track, uint iPart = 1 )
    { fill(&track, iPart); }

    virtual void fillCollection( const std::vector<reco::RecoChargedCandidate> & coll );

    // Clear ntuple cache
    void clearVec();

  protected:
    PhysVarHisto * h_dxy_  ;    //!<   Track dxy
    PhysVarHisto * h_dz_   ;    //!<   Track dsz
    PhysVarHisto * h_nValid_;   //!<   Number of valid hits
    PhysVarHisto * h_nLost_;    //!<   Number of lost hits
  };
}
#endif
