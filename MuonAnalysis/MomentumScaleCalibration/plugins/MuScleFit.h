#ifndef MuScleFit_H
#define MuScleFit_H

/** \class MuScleFit
 *  Analyzer of the Global muon tracks
 *
 *  $Date: 2008/07/03 10:39:22 $
 *  $Revision: 1.1 $
 *  \author C.Mariotti, S.Bolognesi - INFN Torino / T.Dorigo - INFN Padova
 */

// Base Class Headers
// ------------------
#include "FWCore/Framework/interface/EDLooper.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include <CLHEP/Vector/LorentzVector.h>
#include <vector>
#include "Histograms.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class TFile;
class TH1F;
class TH2F;
class TProfile;
class MuonServiceProxy;
class TTree;
class MuScleFitPlotter;

class MuScleFit: public edm::EDLooper {

 public:
  // Constructor
  // -----------
  MuScleFit (const edm::ParameterSet& pset);

  // Destructor
  // ----------
  virtual ~MuScleFit();

  // Operations
  // ----------
  virtual void beginOfJob (const edm::EventSetup& eventSetup) ;
  virtual void endOfJob() ;

  virtual void startingNewLoop (unsigned int iLoop) ;
  virtual edm::EDLooper::Status endOfLoop (const edm::EventSetup& eventSetup, unsigned int iLoop) ;
  virtual edm::EDLooper::Status duringLoop (const edm::Event & event, const edm::EventSetup& eventSetup);

 template<typename T>
  std::vector<reco::LeafCandidate> fillMuonCollection (const std::vector<T>& tracks){
    std::vector<reco::LeafCandidate> muons;
    typename std::vector<T>::const_iterator track;
    for (track = tracks.begin(); track != tracks.end(); ++track){
      reco::Particle::LorentzVector mu(track->px(),track->py(),track->pz(),
				       sqrt(track->p()*track->p() + MuScleFitUtils::mMu2));
	// Apply smearing if needed, and then bias
	// ---------------------------------------
	MuScleFitUtils::goodmuon++;
	if (debug>0) cout <<setprecision(9)<< "Muon #" << MuScleFitUtils::goodmuon 
			  << ": initial value   Pt = " << mu.Pt() << endl;
	if (MuScleFitUtils::SmearType>0) {
	  mu = MuScleFitUtils::applySmearing (mu);
	  if (debug>0) cout << "Muon #" << MuScleFitUtils::goodmuon 
			    << ": after smearing  Pt = " << mu.Pt() << endl;
	} 
	if (MuScleFitUtils::BiasType>0) {
	  mu = MuScleFitUtils::applyBias (mu, track->charge());
	  if (debug>0) cout << "Muon #" << MuScleFitUtils::goodmuon 
			    << ": after bias      Pt = " << mu.Pt() << endl;
	}
	reco::LeafCandidate muon(track->charge(),mu);
	// Store modified muon
	// -------------------
	muons.push_back (muon);
    }
    return muons;
  } 
 protected:

 private:

  //Check if two lorentzVector are near in deltaR
  bool checkDeltaR(reco::Particle::LorentzVector& genMu, reco::Particle::LorentzVector& recMu);

  // Fill, clean and write to file the Map of Histograms
  // ---------------------------------------------------
  void fillHistoMap();
  void clearHistoMap();
  void writeHistoMap();

  MuonServiceProxy *theService;

  std::string theRootFileName;
  std::vector<TFile*> theFiles;

  int debug;

  edm::InputTag theMuonLabel;
  int theMuonType;
  
  // Counters
  // --------
  int numberOfSimTracks;
  int numberOfSimMuons;
  int numberOfSimVertices;
  int numberOfEwkZ;

  // Constants
  // ---------
  double minResMass_sim[6];
  double maxResMass_sim[6];
  double minResMass_rec[6];
  double maxResMass_rec[6];
  double minResMass_hwindow[6]; 
  double maxResMass_hwindow[6];

  // Total number of loops
  // ---------------------
  unsigned int maxLoopNumber;
  unsigned int loopCounter;

  // Tree with muon info for Likelihood evaluation
  // TTree* muonTree;

  // The map of histograms
  // ---------------------
  std::map<std::string, Histograms*> mapHisto;
  MuScleFitPlotter *plotter;

  // The reconstructed muon 4-momenta to be put in the tree
  // ------------------------------------------------------
  reco::Particle::LorentzVector recMu1, recMu2;
  int iev;

  TProfile * Mass_P;
  TProfile * Mass_fine_P;
};
#endif

 
