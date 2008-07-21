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
		 double pt1=0, double pt2=200, double m1=0, double m2=200,
		 TFileDirectory * parentDir=0 );
    virtual ~HistoPhoton();


    // fill a plain ol' photon:
    virtual void fill( const Photon *photon, uint iPart = 1, double weight = 1.0);
    virtual void fill( const Photon &photon, uint iPart = 1, double weight = 1.0 ) { fill(&photon, iPart,weight); }

    // fill a photon that is a shallow clone, and take kinematics from 
    // shallow clone but detector plots from the photon itself
    virtual void fill( const reco::ShallowClonePtrCandidate *photon, uint iPart = 1, double weight = 1.0 );
    virtual void fill( const reco::ShallowClonePtrCandidate &photon, uint iPart = 1, double weight = 1.0 )
    { fill(&photon, iPart, weight); }

    virtual void fillCollection( const std::vector<Photon> & coll, double weight = 1.0 );

    // Clear ntuple cache
    void clearVec();
  protected:

    float  phiNormalization( float& a);


    PhysVarHisto *    h_trackIso_;
    PhysVarHisto *    h_caloIso_;


    PhysVarHisto *    p_nTrackIsol_;
    PhysVarHisto *    p_trackPtSum_;
    PhysVarHisto *    p_ecalSum_;
    PhysVarHisto *    p_hcalSum_;

    PhysVarHisto *    h_nPho_[2][3];
    PhysVarHisto *    h_scEta_[2];
    PhysVarHisto *    h_scPhi_[2];
    PhysVarHisto *    h_scEtaPhi_[2];

    PhysVarHisto *    h_scE_[2][3];
    PhysVarHisto *    h_scEt_[2][3];

    PhysVarHisto *    h_r9_[2][3];
    PhysVarHisto *    h_phoE_[2][3];
    PhysVarHisto *    h_phoEt_[2][3];
    PhysVarHisto *    h_phoEta_[2];
    PhysVarHisto *    h_phoPhi_[2];

//  conversion infos

    PhysVarHisto *    h_nConv_[2][3];
    PhysVarHisto *    h_convEta_[2];
    PhysVarHisto *    h_convPhi_[2];
    PhysVarHisto *    h_r9VsNofTracks_[2][3];
    PhysVarHisto *    h_EoverPTracks_[2][3];
    PhysVarHisto *    p_tk_nHitsVsEta_[2];
    PhysVarHisto *    h_tkChi2_[2];
    PhysVarHisto *    h_DPhiTracksAtVtx_[2][3];
    PhysVarHisto *    h_DCotTracks_[2][3];
    PhysVarHisto *    h_invMass_[2][3];
    PhysVarHisto *    h_DPhiTracksAtEcal_[2][3];
    PhysVarHisto *    h_DEtaTracksAtEcal_[2][3];

    PhysVarHisto *    h_convVtxRvsZ_[2];
    PhysVarHisto *    h_zPVFromTracks_[2];

  };

}
#endif
