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
#include "PhysicsTools/StarterKit/interface/HistoTau.h"
#include "PhysicsTools/StarterKit/interface/HistoJet.h"
#include "PhysicsTools/StarterKit/interface/HistoMET.h"
#include "PhysicsTools/StarterKit/interface/HistoPhoton.h"
#include "PhysicsTools/StarterKit/interface/HistoTrack.h"
#include "PhysicsTools/StarterKit/interface/HistoParticle.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/Candidate/interface/ShallowClonePtrCandidate.h"
#include "DataFormats/Common/interface/Ptr.h"

// STL include files
#include <string>

// ROOT include files
#include <TH1D.h>
#include <TFile.h>

namespace pat {


  template <class T>
  class HistoMap {
  public:
    typedef std::string                              key_type;
    typedef T *                                      data_type;
    typedef std::map<key_type, data_type >           map_type;
    
    map_type map;
  };

  class HistoComposite : public HistoGroup<reco::CompositeCandidate> {

   public:


    typedef edm::Ptr<pat::Muon>        MuonPtr;
    typedef edm::Ptr<pat::Electron>    ElectronPtr;
    typedef edm::Ptr<pat::Tau>         TauPtr;
    typedef edm::Ptr<pat::Photon>      PhotonPtr;
    typedef edm::Ptr<pat::Jet>         JetPtr;
    typedef edm::Ptr<pat::MET>         METPtr;
    
    
    HistoComposite(std::string dir, std::string candTitle, std::string candName,
		   double pt1=0, double pt2=200, double m1=0, double m2=200);
    virtual ~HistoComposite();

    // void fill( reco::CompositeCandidate * cand );
    void fill( const reco::CompositeCandidate * cand );
    void fill( const reco::CompositeCandidate & cand ) { return fill(&cand); }

   protected:
    std::string       candName_;


    HistoMap<HistoMuon>        histoMuon_;
    HistoMap<HistoElectron>    histoElectron_;
    HistoMap<HistoTau>         histoTau_;
    HistoMap<HistoJet>         histoJet_;
    HistoMap<HistoMET>         histoMET_;
    HistoMap<HistoPhoton>      histoPhoton_;
    HistoMap<HistoTrack>       histoTrack_;
    HistoMap<HistoParticle>    histoParticle_;
    HistoMap<HistoComposite>   histoComposite_;
  };


}
#endif
