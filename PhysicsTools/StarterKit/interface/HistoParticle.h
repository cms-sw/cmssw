#ifndef StarterKit_HistoParticle_h
#define StarterKit_HistoParticle_h

//------------------------------------------------------------
// Title: HistoParticle.h
// Purpose: To histogram Particles
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
//   HistoParticle ( TFile * file );
//   Description: Constructor.
//
//   void fill( TK::Particle * );
//   Description: Fill object. Will fill relevant jet variables
//
//   void write();
//   Description: Write object to file in question.
//
//   ~HistoParticle
//    Description: Destructor. Deallocates memory.
//
//------------------------------------------------------------
//
// Modification History:
//
//   -29Nov07: Sal Rappoccio: Creation of the object
//------------------------------------------------------------


// CMSSW include files
#include "DataFormats/PatCandidates/interface/Particle.h"
#include "PhysicsTools/StarterKit/interface/HistoGroup.h"

// STL include files
#include <string>

// ROOT include files
#include <TH1D.h>
#include <TFile.h>

namespace pat {

  class HistoParticle : public HistoGroup<Particle> {

  public:
    HistoParticle( std::string subDir );
    virtual ~HistoParticle();

    void fill( const Particle * part, double weight = 1.0 );
    void fill( const Particle & part, double weight = 1.0 ) { return fill(&part, weight); }


    // Clear ntuple cache
    void clearVec();
  };

}
#endif
