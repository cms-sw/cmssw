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
    HistoMET( std::string dir = "met", std::string group = "MET",
	      std::string pre = "met",
	      double pt1=0, double pt2=200, double m1=0, double m2=200 );
    virtual ~HistoMET();

    // fill a plain ol' met:
    virtual void fill( const MET *met, uint iPart = 1, double weight = 1.0 );
    virtual void fill( const MET &met, uint iPart = 1, double weight = 1.0 ) { fill(&met, iPart, weight); }

    // fill a met that is a shallow clone, and take kinematics from 
    // shallow clone but detector plots from the met itself
    virtual void fill( const reco::ShallowClonePtrCandidate *met, uint iPart = 1, double weight = 1.0 );
    virtual void fill( const reco::ShallowClonePtrCandidate &met, uint iPart = 1, double weight = 1.0 )
    { fill(&met, iPart,weight); }

    virtual void fillCollection( const std::vector<MET> & coll, double weight = 1.0 );

    // Clear ntuple cache
    void clearVec();

  protected:
    
    PhysVarHisto * h_sumEt_;
    PhysVarHisto * h_mEtSig_;
    PhysVarHisto * h_eLongitudinal_;

    PhysVarHisto * h_maxEtInEmTowers_;     
    PhysVarHisto * h_maxEtInHadTowers_;    
    PhysVarHisto * h_etFractionHadronic_; 
    PhysVarHisto * h_emEtFraction_;        
    PhysVarHisto * h_hadEtInHB_;           
    PhysVarHisto * h_hadEtInHO_;           
    PhysVarHisto * h_hadEtInHE_;           
    PhysVarHisto * h_hadEtInHF_;           
    PhysVarHisto * h_emEtInEB_;            
    PhysVarHisto * h_emEtInEE_;            
    PhysVarHisto * h_emEtInHF_;            

  };

}
#endif
