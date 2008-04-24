#ifndef StarterKit_HistoPhoton_h
#define StarterKit_HistoPhoton_h

//------------------------------------------------------------
// Title: HistoPhoton.h
// Purpose: To histogram Photons
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
//   HistoPhoton ( TFile * file );
//   Description: Constructor.
//
//   void fill( TK::Photon * );
//   Description: Fill object. Will fill relevant photon variables
//
//   void write();
//   Description: Write object to file in question.
//
//   ~HistoPhoton
//    Description: Destructor. Deallocates memory.
//
//------------------------------------------------------------
//
// Modification History:
//
//   -29Nov07: Sal Rappoccio: Creation of the object
//------------------------------------------------------------


// CMSSW include files
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "PhysicsTools/StarterKit/interface/HistoGroup.h"

// STL include files
#include <string>

// ROOT include files
#include <TH1D.h>
#include <TFile.h>

namespace pat {

  class HistoPhoton : public HistoGroup<Photon> {

  public:
    HistoPhoton( std::string dir = "photon",std::string group = "Photon",std::string pre="photon",
		   double pt1=0, double pt2=200, double m1=0, double m2=200 );
    virtual ~HistoPhoton();


    // fill a plain ol' photon:
    virtual void fill( const Photon *photon, uint iPart = 1 );
    virtual void fill( const Photon &photon, uint iPart = 1 ) { fill(&photon, iPart); }

    // fill a photon that is a shallow clone, and take kinematics from 
    // shallow clone but detector plots from the photon itself
    virtual void fill( const reco::ShallowCloneCandidate *photon, uint iPart = 1 );
    virtual void fill( const reco::ShallowCloneCandidate &photon, uint iPart = 1 )
    { fill(&photon, iPart); }

    virtual void fillCollection( const std::vector<Photon> & coll );

    // Clear ntuple cache
    void clearVec();
  protected:

    PhysVarHisto *    h_trackIso_;
    PhysVarHisto *    h_caloIso_;
    PhysVarHisto *    h_photonID_;

  };

}
#endif
