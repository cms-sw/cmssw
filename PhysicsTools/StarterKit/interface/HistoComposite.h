#ifndef StarterKit_HistoComposite_h
#define StarterKit_HistoComposite_h

//------------------------------------------------------------
// Title: HistoComposite.h
// Purpose: To histogram Composites
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
//   HistoComposite ( TFile * file );
//   Description: Constructor.
//
//   void fill( TK::Composite * );
//   Description: Fill object. Will fill relevant muon variables
//
//   void write();
//   Description: Write object to file in question.
//
//   ~HistoComposite
//    Description: Destructor. Deallocates memory.
//
//------------------------------------------------------------
//
// Modification History:
//
//   -29Nov07: Sal Rappoccio: Creation of the object
//------------------------------------------------------------


// CMSSW include files
#include "PhysicsTools/StarterKit/interface/HistoMuon.h"
#include "PhysicsTools/StarterKit/interface/HistoElectron.h"
#include "PhysicsTools/StarterKit/interface/HistoJet.h"
#include "PhysicsTools/StarterKit/interface/HistoMET.h"
#include "PhysicsTools/StarterKit/interface/HistoParticle.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"

// STL include files
#include <string>

// ROOT include files
#include <TH1D.h>
#include <TFile.h>

namespace pat {

  class HistoComposite : public HistoGroup<reco::CompositeCandidate> {

   public:
    HistoComposite(std::string dir, std::string candTitle, std::string candName );
    virtual ~HistoComposite();

    // void fill( reco::CompositeCandidate * cand );
    void fill( const reco::CompositeCandidate * cand );
    void fill( const reco::CompositeCandidate & cand ) { return fill(&cand); }

   protected:
    std::string       candName_;


    HistoMuon *     histoMuon_;
    HistoElectron * histoElectron_;
    HistoJet *      histoJet_;
    HistoMET *      histoMET_;
    HistoParticle * histoParticle_;
  };


}
#endif
