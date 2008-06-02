#ifndef StarterKit_PhysicsHistograms_h
#define StarterKit_PhysicsHistograms_h

//------------------------------------------------------------------------
//!  \class PhysicsHistograms
//!  \brief Object to manage and fill various physics histograms
//!
//!  The order how the operations must be executed.
//!
//!  1. we first build our own default histogram groups (electrons, muons, etc)
//!
//!  2. the user-defined histogram groups are added by add*HistoGroup() methods.
//!
//!  3. configure starts:
//!     all PhysVarHisto pointers are collected in one big flat array for
//!     easy access and speedy processing.
//!
//!  4. various histograms are disabled.
//!
//!  5. various histograms are enabled.  configure ends.
//!
//!  At this point we're good to go and ready to see the events.
//------------------------------------------------------------------------



// system include files
#include <memory>
#include <fstream>

// user include files
#include "PhysicsTools/StarterKit/interface/HistoMuon.h"
#include "PhysicsTools/StarterKit/interface/HistoElectron.h"
#include "PhysicsTools/StarterKit/interface/HistoTau.h"
#include "PhysicsTools/StarterKit/interface/HistoJet.h"
#include "PhysicsTools/StarterKit/interface/HistoMET.h"
#include "PhysicsTools/StarterKit/interface/HistoPhoton.h"
#include "PhysicsTools/StarterKit/interface/HistoTrack.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "PhysicsTools/UtilAlgos/interface/TFileDirectory.h"

//
//--- Class declaration.
//

class PhysicsHistograms  {
public:
  explicit PhysicsHistograms ();
  virtual ~PhysicsHistograms();

  //--- Standard methods used in the event processing, called either by ED analyzer
  //    (from the methods of the same name), or by the FWLite macro which does the
  //    event loop).
  //
  virtual void beginJob();  //!<  initialize before seeing any events
  virtual void   endJob();  //!<  do whatever is needed after seeing all events


  //--- Configuration.
  virtual void configure( std::string & histos_to_disable,   // comma separated list of names
			  std::string & histos_to_enable );  // comma separated list of names


  //--- Selection of a subset of PhysVarHistos.
  virtual void select( std::string  vars_to_select,   // comma separated list of names
		       std::vector< pat::PhysVarHisto * > & selectedHistos );

  //--- Clear cache vector for PhysVarHisto
  virtual void clearVec();

  //--- Specific actions for the event.
  // &&& Design note: we could have used overloaded fill() everywhere, but
  // &&&              the novices may find it confusing.

  //--- Process a whole collection of Muons...
  //
  inline void fillCollection( const std::vector<pat::Muon> & coll )
    { muonHistograms_->fillCollection(coll); }

  //--- ... or Electrons...
  //
  inline void fillCollection( const std::vector<pat::Electron> & coll )
    { electronHistograms_->fillCollection(coll); }

  //--- ... or Taus...
  //
  inline void fillCollection( const std::vector<pat::Tau> & coll )
    { tauHistograms_->fillCollection(coll); }

  //--- ... or Jets...
  //
  inline void fillCollection( const std::vector<pat::Jet> & coll )
    { jetHistograms_->fillCollection(coll); }

  //--- ... or MET.
  //
  inline void fillCollection( const std::vector<pat::MET> & coll )
    { metHistograms_->fillCollection(coll); }

  //--- ... or Photon.
  //
  inline void fillCollection( const std::vector<pat::Photon> & coll )
    { photonHistograms_->fillCollection(coll); }

  //--- ... or Track.
  //
  inline void fillCollection( const std::vector<reco::RecoChargedCandidate> & coll )
    { trackHistograms_->fillCollection(coll); }



  // &&& Design note: again, let's be explicit.  This could be compressed into
  // &&&              fewer functions, but at the expense of more complicated
  // &&&              code under the hood, and also an interface which is a teeny
  // &&&              harder to master (and we are trying to avoid that; the
  // &&&              interface should be as dumb as possible).

  //--- Add one histo to muon group, or a whole group of muon histograms
  //
  inline void addMuonHisto ( pat::PhysVarHisto * h )
    { muonHistograms_->addHisto(h); }
  inline void addMuonHistoGroup( pat::HistoMuon * hgr )
    { muonHistograms_->addHistoGroup(hgr); }

  //--- Add one histo to electron group, or a whole group of electron histograms
  //
  inline void addElectronHisto ( pat::PhysVarHisto * h )
    { electronHistograms_->addHisto(h); }
  inline void addElectronHistoGroup( pat::HistoElectron * hgr )
    { electronHistograms_->addHistoGroup(hgr); }

  //--- Add one histo to tau group, or a whole group of tau histograms
  //
  inline void addTauHisto ( pat::PhysVarHisto * h )
    { tauHistograms_->addHisto(h); }
  inline void addTauHistoGroup( pat::HistoTau * hgr )
    { tauHistograms_->addHistoGroup(hgr); }

  //--- Add one histo to jet group, or a whole group of jet histograms
  //
  inline void addJetHisto ( pat::PhysVarHisto * h )
    { jetHistograms_->addHisto(h); }
  inline void addJetHistoGroup( pat::HistoJet * hgr )
    { jetHistograms_->addHistoGroup(hgr); }

  //--- Add one histo to MET group, or a whole group of MET histograms
  //
  inline void addMetHisto ( pat::PhysVarHisto * h )
    { metHistograms_->addHisto(h); }
  inline void addMetHistoGroup( pat::HistoMET * hgr )
    { metHistograms_->addHistoGroup(hgr); }

  //--- Add one histo to photon group, or a whole group of photon histograms
  //
  inline void addPhotonHisto ( pat::PhysVarHisto * h )
    { photonHistograms_->addHisto(h); }
  inline void addPhotonHistoGroup( pat::HistoPhoton * hgr )
    { photonHistograms_->addHistoGroup(hgr); }


  //--- Add one histo to track group, or a whole group of track histograms
  //
  inline void addTrackHisto ( pat::PhysVarHisto * h )
    { trackHistograms_->addHisto(h); }
  inline void addTrackHistoGroup( pat::HistoTrack * hgr )
    { trackHistograms_->addHistoGroup(hgr); }

  //--- Add one generic histo to list
  inline void addHisto( pat::PhysVarHisto * h )
    { allVarHistos_.push_back( h ); }



private:

  // Parameters for running
  std::string     outputTextName_;

  // Histogram server
  edm::Service<TFileService> fs;

  // Histogram objects that make "standard" plots for each object
  pat::HistoMuon     * muonHistograms_;
  pat::HistoElectron * electronHistograms_;
  pat::HistoTau      * tauHistograms_;
  pat::HistoMET      * metHistograms_;
  pat::HistoJet      * jetHistograms_;
  pat::HistoPhoton   * photonHistograms_;
  pat::HistoTrack    * trackHistograms_;

  //--- The summary of all PhysVarHistos.
  // &&& Is this still needed?
  std::vector< pat::PhysVarHisto* > allVarHistos_ ;
  std::vector< pat::PhysVarHisto* > enabledVarHistos_ ;

  //--- This is a nice feature but let's not worry about it for now. &&&
  ofstream        outputFile_;
};

#endif
