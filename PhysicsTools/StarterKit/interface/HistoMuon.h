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
	      double pt1=0, double pt2=200, double m1=0, double m2=200,
	      TFileDirectory * parentDir=0);
    virtual ~HistoMuon() { } ;

    // fill a plain ol' muon:
    virtual void fill( const Muon *muon, uint iPart = 1, double weight = 1.0 );
    virtual void fill( const Muon &muon, uint iPart = 1, double weight = 1.0 ) { fill(&muon, iPart,weight); }

    // fill a muon that is a shallow clone, and take kinematics from 
    // shallow clone but detector plots from the muon itself
    virtual void fill( const reco::ShallowClonePtrCandidate *muon, uint iPart = 1, double weight = 1.0 );
    virtual void fill( const reco::ShallowClonePtrCandidate &muon, uint iPart = 1, double weight = 1.0 )
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

//muon energy deposit analyzer
    PhysVarHisto * ecalDepEnergy_;
    PhysVarHisto * ecalS9DepEnergy_ ;
    PhysVarHisto * hcalDepEnergy_ ;
    PhysVarHisto * hcalS9DepEnergy_ ;
    PhysVarHisto * hoDepEnergy_ ;
    PhysVarHisto * hoS9DepEnergy_ ;
/*
// muon seed analyzer
    PhysVarHisto * NumberOfRecHitsPerSeed_ ;
    PhysVarHisto * seedPhi_ ;
    PhysVarHisto * seedEta_ ;
    PhysVarHisto * seedTheta_ ;
    PhysVarHisto * seedPt_ ;
    PhysVarHisto * seedPx_ ;
    PhysVarHisto * seedPy_ ;
    PhysVarHisto * seedPz_ ;
    PhysVarHisto * seedPtErr_ ;
    PhysVarHisto * seedPtErrVsPhi_ ;
    PhysVarHisto * seedPtErrVsEta_ ;
    PhysVarHisto * seedPtErrVsPt_ ;
    PhysVarHisto * seedPxErr_ ;
    PhysVarHisto * seedPyErr_ ;
    PhysVarHisto * seedPzErr_ ;
    PhysVarHisto * seedPErr_ ;
    PhysVarHisto * seedPErrVsPhi_ ;
    PhysVarHisto * seedPErrVsEta_ ;
    PhysVarHisto * seedPErrVsPt_ ;
    PhysVarHisto * seedPhiErr_ ;
    PhysVarHisto * seedEtaErr_ ;
*/
// muon reco analyzer
    PhysVarHisto * muReco_ ;
// global muon
    std::vector<PhysVarHisto *> etaGlbTrack_ ;
    std::vector<PhysVarHisto *> etaResolution_ ;
    std::vector<PhysVarHisto *> thetaGlbTrack_ ;
    std::vector<PhysVarHisto *> thetaResolution_ ;
    std::vector<PhysVarHisto *> phiGlbTrack_ ;
    std::vector<PhysVarHisto *> phiResolution_ ;
    std::vector<PhysVarHisto *> pGlbTrack_ ;
    std::vector<PhysVarHisto *> ptGlbTrack_ ;
    std::vector<PhysVarHisto *> qGlbTrack_ ;
    std::vector<PhysVarHisto *> qOverpResolution_ ;
    std::vector<PhysVarHisto *> qOverptResolution_ ;
    std::vector<PhysVarHisto *> oneOverpResolution_ ;
    std::vector<PhysVarHisto *> oneOverptResolution_ ;

// tracker muon
    PhysVarHisto * etaTrack_ ;
    PhysVarHisto * thetaTrack_ ;
    PhysVarHisto * phiTrack_ ;
    PhysVarHisto * pTrack_ ;
    PhysVarHisto * ptTrack_ ;
    PhysVarHisto * qTrack_ ;
//sta muon
    PhysVarHisto * etaStaTrack_ ;
    PhysVarHisto * thetaStaTrack_ ;
    PhysVarHisto * phiStaTrack_ ;
    PhysVarHisto * pStaTrack_ ;
    PhysVarHisto * ptStaTrack_ ;
    PhysVarHisto * qStaTrack_ ;

// segment track analyzer
// GlbTrack
    PhysVarHisto * GlbhitsNotUsed_ ;
    PhysVarHisto * GlbhitsNotUsedPercentual_ ;
    PhysVarHisto * GlbTrackSegm_ ;
    PhysVarHisto * GlbhitStaProvenance_ ;
    PhysVarHisto * GlbhitTkrProvenance_ ;
    PhysVarHisto * GlbtrackHitPercentualVsEta_ ;
    PhysVarHisto * GlbtrackHitPercentualVsPhi_ ;
    PhysVarHisto * GlbtrackHitPercentualVsPt_ ;
    PhysVarHisto * GlbdtTrackHitPercentualVsEta_ ;
    PhysVarHisto * GlbdtTrackHitPercentualVsPhi_ ;
    PhysVarHisto * GlbdtTrackHitPercentualVsPt_ ;
    PhysVarHisto * GlbcscTrackHitPercentualVsEta_ ;
    PhysVarHisto * GlbcscTrackHitPercentualVsPhi_ ;
    PhysVarHisto * GlbcscTrackHitPercentualVsPt_ ;


// StandAlone Muon
    PhysVarHisto * StahitsNotUsed_ ;
    PhysVarHisto * StahitsNotUsedPercentual_ ;
    PhysVarHisto * StaTrackSegm_ ;
    PhysVarHisto * StahitStaProvenance_ ;
    PhysVarHisto * StahitTkrProvenance_ ;
    PhysVarHisto * StatrackHitPercentualVsEta_ ;
    PhysVarHisto * StatrackHitPercentualVsPhi_ ;
    PhysVarHisto * StatrackHitPercentualVsPt_ ;
    PhysVarHisto * StadtTrackHitPercentualVsEta_ ;
    PhysVarHisto * StadtTrackHitPercentualVsPhi_ ;
    PhysVarHisto * StadtTrackHitPercentualVsPt_ ;
    PhysVarHisto * StacscTrackHitPercentualVsEta_ ;
    PhysVarHisto * StacscTrackHitPercentualVsPhi_ ;
    PhysVarHisto * StacscTrackHitPercentualVsPt_ ;



  };
}
#endif
