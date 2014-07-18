// #define DEBUG
// System include files
// --------------------

// User include files
// ------------------
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include <CLHEP/Vector/LorentzVector.h>

// For file output
// ---------------
#include <fstream>
#include <sstream>
#include <cmath>
#include <memory>

#include <vector>

#include "TRandom.h"

// Class declaration
// -----------------

class MuScleFitFilter : public edm::EDFilter {
 public:
  explicit MuScleFitFilter(const edm::ParameterSet&);
  ~MuScleFitFilter();

 private:
  virtual bool filter(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override {};

  // Member data
  // -----------
  int eventsRead;
  int eventsWritten;
  bool debug;
  int theMuonType;
  std::vector<double> Mmin;
  std::vector<double> Mmax;
  int maxWrite;
  unsigned int minimumMuonsNumber;

  // Collections labels
  // ------------------
  edm::InputTag theMuonLabel;
  edm::EDGetTokenT<reco::MuonCollection> theGlbMuonsToken;
  edm::EDGetTokenT<reco::TrackCollection> theSaMuonsToken;
  edm::EDGetTokenT<reco::TrackCollection> theTracksToken;

};

// Static data member definitions
// ------------------------------
const double Mmu2 = 0.011163612;    // Squared muon mass

// Constructor
// -----------
MuScleFitFilter::MuScleFitFilter(const edm::ParameterSet& iConfig)
{
  debug = iConfig.getUntrackedParameter<bool>("debug",false);

  if (debug)
    std::cout << "Constructor" << std::endl;

  // Parameters
  // ----------
  //ParameterSet serviceParameters = iConfig.getParameter<ParameterSet>("ServiceParameters");
  //theService = new MuonServiceProxy(serviceParameters);
  theMuonLabel = iConfig.getParameter<edm::InputTag>("MuonLabel");
  theGlbMuonsToken = mayConsume<reco::MuonCollection>(theMuonLabel);
  theSaMuonsToken = mayConsume<reco::TrackCollection>(theMuonLabel);
  theTracksToken = mayConsume<reco::TrackCollection>(theMuonLabel);
  theMuonType = iConfig.getParameter<int>("muonType");

  Mmin = iConfig.getUntrackedParameter<std::vector<double> >("Mmin");
  Mmax = iConfig.getUntrackedParameter<std::vector<double> >("Mmax");
  maxWrite = iConfig.getUntrackedParameter<int>("maxWrite",100000);

  minimumMuonsNumber = iConfig.getUntrackedParameter<unsigned int>("minimumMuonsNumber", 2);

  // The must have the same size and they must not be empty, otherwise abort.
  if ( !(Mmin.size() == Mmax.size() && !Mmin.empty()) ) abort();

  // Count the total number of analyzed and written events
  // -----------------------------------------------------
  eventsRead = 0;
  eventsWritten = 0;

}


// Destructor
// ----------
MuScleFitFilter::~MuScleFitFilter() {
  std::cout << "Total number of events read    = " << eventsRead << std::endl;
  std::cout << "Total number of events written = " << eventsWritten << std::endl;
}

// Member functions
// ----------------

// Method called for each event
// ----------------------------
bool MuScleFitFilter::filter(edm::Event& event, const edm::EventSetup& iSetup) {

  // Cut the crap if we have stored enough stuff
  // -------------------------------------------
  if ( maxWrite != -1 && eventsWritten>=maxWrite ) return false;

  // Get the RecTrack and the RecMuon collection from the event
  // ----------------------------------------------------------
  std::auto_ptr<reco::MuonCollection> muons(new reco::MuonCollection());

  if (debug) std::cout << "Looking for muons of the right kind" << std::endl;

  if (theMuonType==1) { // GlobalMuons

    // Global muons:
    // -------------
    edm::Handle<reco::MuonCollection> glbMuons;
    if (debug) std::cout << "Handle defined" << std::endl;
    event.getByToken(theGlbMuonsToken, glbMuons);
    if (debug)
      std::cout << "Global muons: " << glbMuons->size() << std::endl;

    // Store the muon
    // --------------
    reco::MuonCollection::const_iterator glbMuon;
    for (glbMuon=glbMuons->begin(); glbMuon!=glbMuons->end(); ++glbMuon) {
      muons->push_back(*glbMuon);
      if (debug) {
	std::cout << "  Reconstructed muon: pT = " << glbMuon->p4().Pt()
	     << "  Eta = " << glbMuon->p4().Eta() << std::endl;
      }
    }
  } else if (theMuonType==2) { // StandaloneMuons

    // Standalone muons:
    // -----------------
    edm::Handle<reco::TrackCollection> saMuons;
    event.getByToken(theSaMuonsToken, saMuons);
    if (debug)
      std::cout << "Standalone muons: " << saMuons->size() << std::endl;

    // Store the muon
    // --------------
    reco::TrackCollection::const_iterator saMuon;
    for (saMuon=saMuons->begin(); saMuon!=saMuons->end(); ++saMuon) {
      reco::Muon muon;
      double energy = sqrt(saMuon->p()*saMuon->p()+Mmu2);
      math::XYZTLorentzVector p4(saMuon->px(), saMuon->py(), saMuon->pz(), energy);
      muon.setP4(p4);
      muon.setCharge(saMuon->charge());
      muons->push_back(muon);
    }
  } else if (theMuonType==3) { // Tracker tracks

    // Tracks:
    // -------
    edm::Handle<reco::TrackCollection> tracks;
    event.getByToken(theTracksToken, tracks);
    if (debug)
      std::cout << "Tracker tracks: " << tracks->size() << std::endl;

    // Store the muon
    // -------------
    reco::TrackCollection::const_iterator track;
    for (track=tracks->begin(); track!=tracks->end(); ++track) {
      reco::Muon muon;
      double energy = sqrt(track->p()*track->p()+Mmu2);
      math::XYZTLorentzVector p4(track->px(), track->py(), track->pz(), energy);
      muon.setP4(p4);
      muon.setCharge(track->charge());
      muons->push_back(muon);
    }
  } else {
    std::cout << "Wrong muon type! Aborting." << std::endl;
    abort();
  }

  // Loop on RecMuon and reconstruct the resonance
  // ---------------------------------------------
  reco::MuonCollection::const_iterator muon1;
  reco::MuonCollection::const_iterator muon2;

  bool resfound = false;

  // Require at least N muons of the selected type.
  if( muons->size() >= minimumMuonsNumber ) {

    for (muon1=muons->begin(); muon1!=muons->end(); ++muon1) {

      if (debug) {
        std::cout << "  Reconstructed muon: pT = " << muon1->p4().Pt()
             << "  Eta = " << muon1->p4().Eta() << std::endl;
      }

      // Recombine all the possible Z from reconstructed muons
      // -----------------------------------------------------
      if (muons->size()>1) {
        for (muon2 = muon1+1; muon2!=muons->end(); ++muon2) {
          if ( ((*muon1).charge()*(*muon2).charge())<0 ) { // This also gets rid of muon1==muon2
            //	  reco::Particle::LorentzVector Z (muonCorr1 + muonCorr2);
            reco::Particle::LorentzVector Z (muon1->p4()+muon2->p4());
            // Loop on all the cuts on invariant mass.
            // If it passes at least one of the cuts, the event will be accepted.
            // ------------------------------------------------------------------
            std::vector<double>::const_iterator mMinCut = Mmin.begin();
            std::vector<double>::const_iterator mMaxCut = Mmax.begin();
            for( ; mMinCut != Mmin.end(); ++mMinCut, ++mMaxCut ) {
              // When the two borders are -1 do not cut.
              if( *mMinCut == *mMaxCut && *mMaxCut == -1) {
                resfound = true;
                if (debug) {
                  std::cout << "Acceptiong event because mMinCut = " << *mMinCut << " = mMaxCut = " << *mMaxCut << std::endl;
                }
              }
              else if (Z.mass()>*mMinCut && Z.mass()<*mMaxCut) {
                resfound = true;
                if (debug) {
                  std::cout << "One particle found with mass = " << Z.mass() << std::endl;
                }
              }
            }
          }
        }
      } else if (debug) {
        std::cout << "Not enough reconstructed muons to make a resonance" << std::endl;
      }
    }
  }
  else if (debug) {
    std::cout << "Skipping event because muons = " << muons->size() << " < " << "minimumMuonsNumber("<<minimumMuonsNumber<<")" << std::endl;
  }

  // Store the event if it has a dimuon pair with mass within defined boundaries
  // ---------------------------------------------------------------------------
  bool write = false;
  eventsRead++;
  if ( resfound ) {
    write = true;
    eventsWritten++;
  }
  return write;

}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MuScleFitFilter);
