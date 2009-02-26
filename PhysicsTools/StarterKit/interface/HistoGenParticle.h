#ifndef StarterKit_HistoGenParticle_h
#define StarterKit_HistoGenParticle_h

//------------------------------------------------------------
// Title: HistoGenParticle.h
// Purpose: To histogram GenParticles
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
//   HistoGenParticle ( TFile * file );
//   Description: Constructor.
//
//   void fill( TK::GenParticle * );
//   Description: Fill object. Will fill relevant jet variables
//
//   void write();
//   Description: Write object to file in question.
//
//   ~HistoGenParticle
//    Description: Destructor. Deallocates memory.
//
//------------------------------------------------------------
//
// Modification History:
//
//   -29Nov07: Sal Rappoccio: Creation of the object
//------------------------------------------------------------


// CMSSW include files
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "PhysicsTools/StarterKit/interface/HistoGroup.h"

// STL include files
#include <string>

// ROOT include files
#include <TH1D.h>
#include <TFile.h>

namespace pat {

  class HistoGenParticle : public HistoGroup<reco::GenParticle> {

  public:
    HistoGenParticle(std::string dir = "particle", std::string group = "GenParticle",
	      std::string pre ="p",
	      double pt1=0, double pt2=200, double m1=0, double m2=200,
	      TFileDirectory * parentDir=0);
    virtual ~HistoGenParticle() ;

    // fill a plain ol' particle:
    virtual void fill( const reco::GenParticle *particle, double weight = 1.0 );
    virtual void fill( const reco::GenParticle &particle, double weight = 1.0 ) { fill(&particle, weight); }

    // fill a particle that is a shallow clone, and take kinematics from 
    // shallow clone but detector plots from the particle itself
    virtual void fill( const reco::ShallowClonePtrCandidate *particle, double weight = 1.0 );
    virtual void fill( const reco::ShallowClonePtrCandidate &particle, double weight = 1.0 )
    { fill(&particle, weight); }

    virtual void fillCollection( const std::vector<reco::GenParticle> & coll, double weight = 1.0 );

    // Clear ntuple cache
    void clearVec();

  protected:
    // break down pdg ids into a few categories
    int   getKey( reco::Candidate const * p );

  };

}
#endif
