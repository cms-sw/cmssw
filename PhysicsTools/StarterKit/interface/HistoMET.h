#ifndef StarterKit_HistoMET_h
#define StarterKit_HistoMET_h

//------------------------------------------------------------
// Title: HistoMET.h
// Purpose: To histogram METs
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
//   HistoMET ( TFile * file );
//   Description: Constructor.
//
//   void fill( TK::MET * );
//   Description: Fill object. Will fill relevant jet variables
//
//   void write();
//   Description: Write object to file in question.
//
//   ~HistoMET
//    Description: Destructor. Deallocates memory.
//
//------------------------------------------------------------
//
// Modification History:
//
//   -29Nov07: Sal Rappoccio: Creation of the object
//------------------------------------------------------------


// CMSSW include files
#include "DataFormats/PatCandidates/interface/MET.h"
#include "PhysicsTools/StarterKit/interface/HistoGroup.h"

// STL include files
#include <string>

// ROOT include files
#include <TH1D.h>
#include <TFile.h>

namespace pat {

  class HistoMET : public HistoGroup<MET> {

  public:
    HistoMET( std::string dir = "met",
	      double pt1=0, double pt2=200, double m1=0, double m2=200 );
    virtual ~HistoMET();

    void fill( const MET * met, uint iPart = 0  );
    void fill( const MET & met, uint iPart = 0  ) { fill(&met, iPart); }

    //     // Clear ntuple cache
    //     void clearVec();
  };

}
#endif
