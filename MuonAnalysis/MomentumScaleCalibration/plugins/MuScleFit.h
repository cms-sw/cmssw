#ifndef MuScleFit_H
#define MuScleFit_H

/** \class MuScleFit
 *  Analyzer of the Global muon tracks
 *
 *  $Date: 2010/10/22 17:48:07 $
 *  $Revision: 1.42 $
 *  \author C.Mariotti, S.Bolognesi - INFN Torino / T.Dorigo - INFN Padova
 */

// Base Class Headers
// ------------------
#include "FWCore/Framework/interface/EDLooper.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include <CLHEP/Vector/LorentzVector.h>
#include <vector>
// #include "Histograms.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "MuScleFitBase.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/CaloMuon.h"
// The following is required in CMSSW v2.0.x (was contained in Muon.h in 1.6.7)
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"

#include "MuScleFitMuonSelector.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class TH1F;
class TH2F;
class TProfile;
class MuonServiceProxy;
class TTree;
class MuScleFitPlotter;

class MuScleFit: public edm::EDLooper, MuScleFitBase
{
 public:
  // Constructor
  // -----------
  MuScleFit( const edm::ParameterSet& pset );

  // Destructor
  // ----------
  virtual ~MuScleFit();

  // Operations
  // ----------
  void beginOfJobInConstructor();
  // void beginOfJob( const edm::EventSetup& eventSetup );
  // virtual void beginOfJob();
  virtual void endOfJob();

  virtual void startingNewLoop( unsigned int iLoop );

  virtual edm::EDLooper::Status endOfLoop( const edm::EventSetup& eventSetup, unsigned int iLoop );
  virtual void endOfFastLoop( const unsigned int iLoop );

  virtual edm::EDLooper::Status duringLoop( const edm::Event & event, const edm::EventSetup& eventSetup );
  /**
   * This method performs all needed operations on the muon pair. It reads the muons from SavedPair and uses the iev
   * counter to keep track of the event number. The iev is incremented internally and reset to 0 in startingNewLoop.
   */
  virtual void duringFastLoop();

  template<typename T>
  std::vector<reco::LeafCandidate> fillMuonCollection( const std::vector<T>& tracks );
 private:

 protected:
  /**
   * Selects the muon pairs and fills the SavedPair and (if needed) the genPair vector.
   * This version reads the events from the edm root file and performs a selection of the muons according to the parameters in the cfg.
   */
  void selectMuons(const edm::Event & event);
  /**
   * Selects the muon pairs and fills the SavedPair and (if needed) the genPair vector.
   * This version reads the events from a tree in the file specified in the cfg. The tree only contains one muon pair per event. This
   * means that no selection is performed and we use preselected muons.
   */
  void selectMuons(const int maxEvents, const TString & treeFileName);

  /// Template method used to fill the track collection starting from reco::muons or pat::muons
  template<typename T>
  void takeSelectedMuonType(const T & muon, std::vector<reco::Track> & tracks);
  /// Function for onia selections
  bool selGlobalMuon(const pat::Muon* aMuon);
  bool selTrackerMuon(const pat::Muon* aMuon);  

  /// Check if two lorentzVector are near in deltaR
  bool checkDeltaR( reco::Particle::LorentzVector& genMu, reco::Particle::LorentzVector& recMu );
  /// Fill the reco vs gen and reco vs sim comparison histograms
  void fillComparisonHistograms( const reco::Particle::LorentzVector & genMu, const reco::Particle::LorentzVector & recoMu, const std::string & inputName, const int charge );

  /// Apply the smearing if needed using the function in MuScleFitUtils
  void applySmearing( reco::Particle::LorentzVector & mu );
  /// Apply the bias if needed using the function in MuScleFitUtils
  void applyBias( reco::Particle::LorentzVector & mu, const int charge );

  /**
   * Simple method to check parameters consistency. It aborts the job if the parameters
   * are not consistent.
   */
  void checkParameters();

  MuonServiceProxy *theService;

  // Counters
  // --------
  int numberOfSimTracks;
  int numberOfSimMuons;
  int numberOfSimVertices;
  int numberOfEwkZ;

  bool ifHepMC;
  bool ifGenPart;

  // Constants
  // ---------
  double minResMass_hwindow[6];
  double maxResMass_hwindow[6];

  // Total number of loops
  // ---------------------
  unsigned int maxLoopNumber;
  unsigned int loopCounter;

  bool fastLoop;

  MuScleFitPlotter *plotter;

  // The reconstructed muon 4-momenta to be put in the tree
  // ------------------------------------------------------
  reco::Particle::LorentzVector recMu1, recMu2;
  int iev;
  int totalEvents_;

  bool compareToSimTracks_;
  edm::InputTag simTracksCollection_;
  bool PATmuons_;
  std::string genParticlesName_;

  // Input Root Tree file name. If empty events are read from the edm root file.
  std::string inputRootTreeFileName_;
  // Output Root Tree file name. If not empty events are dumped to this file at the end of the last iteration.
  std::string outputRootTreeFileName_;
  // Maximum number of events from root tree. It works in the same way as the maxEvents to configure a input source.
  int maxEventsFromRootTree_;

  std::string triggerResultsLabel_;
  std::string triggerResultsProcess_;
  std::string triggerPath_;
  bool negateTrigger_;
  bool saveAllToTree_;

  std::auto_ptr<MuScleFitMuonSelector> muonSelector_;
};

template<typename T>
std::vector<reco::LeafCandidate> MuScleFit::fillMuonCollection( const std::vector<T>& tracks )
{
  std::vector<reco::LeafCandidate> muons;
  typename std::vector<T>::const_iterator track;
  for( track = tracks.begin(); track != tracks.end(); ++track ) {
    reco::Particle::LorentzVector mu;
    mu = reco::Particle::LorentzVector(track->px(),track->py(),track->pz(),
                                       sqrt(track->p()*track->p() + MuScleFitUtils::mMu2));
    // Apply smearing if needed, and then bias
    // ---------------------------------------
    MuScleFitUtils::goodmuon++;
    if (debug_>0) 
      std::cout <<std::setprecision(9)<< "Muon #" << MuScleFitUtils::goodmuon
                       << ": initial value   Pt = " << mu.Pt() << std::endl;

    applySmearing(mu);
    applyBias(mu, track->charge());

    reco::LeafCandidate muon(track->charge(),mu);
    // Store modified muon
    // -------------------
    muons.push_back (muon);
  }
  return muons;
}

template<typename T>
void MuScleFit::takeSelectedMuonType(const T & muon, std::vector<reco::Track> & tracks)
{
  // std::cout<<"muon "<<muon->isGlobalMuon()<<muon->isStandAloneMuon()<<muon->isTrackerMuon()<<std::endl;
  //NNBB: one muon can be of many kinds at once but with the theMuonType_ we are sure
  // to avoid double counting of the same muon
  if(muon->isGlobalMuon() && theMuonType_==1)
    tracks.push_back(*(muon->globalTrack()));
  else if(muon->isStandAloneMuon() && theMuonType_==2)
    tracks.push_back(*(muon->outerTrack()));
  else if(muon->isTrackerMuon() && theMuonType_==3)
    tracks.push_back(*(muon->innerTrack()));

  else if( theMuonType_ == 10 && !(muon->isStandAloneMuon()) ) //particular case!!
    tracks.push_back(*(muon->innerTrack()));
  else if( theMuonType_ == 11 && muon->isGlobalMuon() )
    tracks.push_back(*(muon->innerTrack()));
  else if( theMuonType_ == 13 && muon->isTrackerMuon() )
    tracks.push_back(*(muon->innerTrack()));
}

#endif
