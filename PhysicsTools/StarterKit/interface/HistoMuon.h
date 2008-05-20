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
    HistoMuon(std::string dir = "muon", std::string group = "Muon",
	      std::string pre ="mu",
		   double pt1=0, double pt2=200, double m1=0, double m2=200 );
    virtual ~HistoMuon() { } ;

    // fill a plain ol' muon:
    virtual void fill( const Muon *muon, uint iPart = 1, double weight = 1.0 );
    virtual void fill( const Muon &muon, uint iPart = 1, double weight = 1.0 ) { fill(&muon, iPart,weight); }

    // fill a muon that is a shallow clone, and take kinematics from 
    // shallow clone but detector plots from the muon itself
    virtual void fill( const reco::ShallowCloneCandidate *muon, uint iPart = 1, double weight = 1.0 );
    virtual void fill( const reco::ShallowCloneCandidate &muon, uint iPart = 1, double weight = 1.0 )
    { fill(&muon, iPart,weight); }

    virtual void fillCollection( const std::vector<Muon> & coll, double weight = 1.0 );

    // Clear ntuple cache
    void clearVec();

  protected:
    PhysVarHisto * h_trackIso_ ;   //!<   &&& document this
    PhysVarHisto * h_caloIso_  ;   //!<   &&& document this
    PhysVarHisto * h_leptonID_ ;   //!<   &&& document this
    PhysVarHisto * h_calCompat_;   //!<   &&& document this
    PhysVarHisto * h_caloE_    ;   //!<   &&& document this
    PhysVarHisto * h_type_     ;   //!<   &&& document this
    PhysVarHisto * h_nChambers_;   //!<   &&& document this
  };
}
#endif
