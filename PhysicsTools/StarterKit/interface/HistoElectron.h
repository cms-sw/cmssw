#ifndef StarterKit_HistoElectron_h
#define StarterKit_HistoElectron_h

//------------------------------------------------------------
// Title: HistoElectron.h
// Purpose: To histogram Electrons
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
//   HistoElectron ( TFile * file );
//   Description: Constructor.
//
//   void fill( TK::Electron * );
//   Description: Fill object. Will fill relevant electron variables
//
//   void write();
//   Description: Write object to file in question.
//
//   ~HistoElectron
//    Description: Destructor. Deallocates memory.
//
//------------------------------------------------------------
//
// Modification History:
//
//   -29Nov07: Sal Rappoccio: Creation of the object
//------------------------------------------------------------


// CMSSW include files
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "PhysicsTools/StarterKit/interface/HistoGroup.h"

// STL include files
#include <string>

// ROOT include files
#include <TH1D.h>
#include <TFile.h>

namespace pat {

  class HistoElectron : public HistoGroup<Electron> {

  public:
    HistoElectron( std::string dir = "electron", std::string group = "Electron",
		   std::string pre = "e",
		   double pt1=0, double pt2=200, double m1=0, double m2=200 ,
		   TFileDirectory * parentDir=0);
    virtual ~HistoElectron();

    // fill a plain ol' electron:
    virtual void fill( const Electron *electron, uint iPart = 1, double weight = 1.0 );
    virtual void fill( const Electron &electron, uint iPart = 1, double weight = 1.0 ) { fill(&electron, iPart,weight); }

    // fill a electron that is a shallow clone, and take kinematics from 
    // shallow clone but detector plots from the electron itself
    virtual void fill( const reco::ShallowClonePtrCandidate *electron, uint iPart = 1, double weight = 1.0 );
    virtual void fill( const reco::ShallowClonePtrCandidate &electron, uint iPart = 1, double weight = 1.0 )
    { fill(&electron, iPart,weight); }

    virtual void fillCollection( const std::vector<Electron> & coll, double weight = 1.0 );

    // Clear ntuple cache
    void clearVec();
  protected:

    PhysVarHisto *    h_trackIso_;
    PhysVarHisto *    h_caloIso_;
    PhysVarHisto *    h_leptonID_;

    PhysVarHisto *    h_ele_matchingObjectEta_;
    PhysVarHisto *    h_ele_matchingObjectPt_;
    PhysVarHisto *    h_ele_matchingObjectPhi_;
    PhysVarHisto *    h_ele_matchingObjectZ_;

    PhysVarHisto *    h_ele_matchingObjectEta_matched_;
    PhysVarHisto *    h_ele_matchingObjectPt_matched_;
    PhysVarHisto *    h_ele_matchingObjectPhi_matched_;
    PhysVarHisto *    h_ele_matchingObjectZ_matched_;

    // electron basic quantities
    PhysVarHisto *    h_ele_vertexP_;
    PhysVarHisto *    h_ele_vertexPt_;
    PhysVarHisto *    h_ele_vertexEta_;
    PhysVarHisto *    h_ele_vertexPhi_;
    PhysVarHisto *    h_ele_vertexX_;
    PhysVarHisto *    h_ele_vertexY_;
    PhysVarHisto *    h_ele_vertexZ_;
    PhysVarHisto *    h_ele_charge_;

    //  electron matching and ID
    PhysVarHisto *    h_ele_EoP_;
    PhysVarHisto *    h_ele_EoPout_;
    PhysVarHisto *    h_ele_dEtaSc_propVtx_;
    PhysVarHisto *    h_ele_dPhiSc_propVtx_;
    PhysVarHisto *    h_ele_dPhiCl_propOut_;
    PhysVarHisto *    h_ele_HoE_;
    PhysVarHisto *    h_ele_PinMnPout_mode_;
    PhysVarHisto *    h_ele_classes_;
    PhysVarHisto *    h_ele_eta_golden_;
    PhysVarHisto *    h_ele_eta_shower_;
    PhysVarHisto *    h_ele_eta_goldenFrac_;
    PhysVarHisto *    h_ele_eta_showerFrac_;


    PhysVarHisto *    h_ele_eta_;


    //electron track
    PhysVarHisto *    h_ele_foundHits_;
    PhysVarHisto *    h_ele_chi2_;

    //efficiencies
    PhysVarHisto *    h_ele_etaEff_;
    PhysVarHisto *    h_ele_ptEff_;
    PhysVarHisto *    h_ele_phiEff_;
    PhysVarHisto *    h_ele_zEff_;



  };

}
#endif
