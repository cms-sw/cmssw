//#define DEBUG

#include "MuScleFitFilter.h"


// Collaborating Class Header
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

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

#include "TRandom.h"

// Namespaces
// ----------
using namespace std;
using namespace edm;

// Static data member definitions
// ------------------------------
const double Mmu2 = 0.011163612;    // Squared muon mass


// Constructor
// -----------
MuScleFitFilter::MuScleFitFilter(const ParameterSet& iConfig) {

  debug = iConfig.getUntrackedParameter<bool>("debug",false);
  
  if (debug)
    cout << "Constructor" << endl;

  // Parameters
  // ----------
  //ParameterSet serviceParameters = iConfig.getParameter<ParameterSet>("ServiceParameters");
  //theService = new MuonServiceProxy(serviceParameters);  
  theMuonLabel = iConfig.getParameter<InputTag>("MuonLabel");
  theMuonType = iConfig.getParameter<int>("muonType");

  Mmin = iConfig.getUntrackedParameter<vector<double> >("Mmin");
  Mmax = iConfig.getUntrackedParameter<vector<double> >("Mmax");
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
  cout << "Total number of events read    = " << eventsRead << endl;
  cout << "Total number of events written = " << eventsWritten << endl;
}

// Member functions
// ----------------

// Method called for each event 
// ----------------------------
bool MuScleFitFilter::filter(Event& event, const EventSetup& iSetup) {

  // Cut the crap if we have stored enough stuff
  // -------------------------------------------
  if ( maxWrite != -1 && eventsWritten>=maxWrite ) return false;

  // Get the RecTrack and the RecMuon collection from the event
  // ----------------------------------------------------------
  std::auto_ptr<reco::MuonCollection> muons(new reco::MuonCollection());

  if (debug) cout << "Looking for muons of the right kind" << endl;

  if (theMuonType==1) { // GlobalMuons

    // Global muons:
    // -------------
    Handle<reco::MuonCollection> glbMuons;
    if (debug) cout << "Handle defined" << endl;
    event.getByLabel(theMuonLabel, glbMuons);
    if (debug)
      cout << "Global muons: " << glbMuons->size() << endl;

    // Store the muon 
    // --------------
    reco::MuonCollection::const_iterator glbMuon;
    for (glbMuon=glbMuons->begin(); glbMuon!=glbMuons->end(); ++glbMuon) {   
      muons->push_back(*glbMuon);
      if (debug) {    
	cout << "  Reconstructed muon: pT = " << glbMuon->p4().Pt()
	     << "  Eta = " << glbMuon->p4().Eta() << endl;
      } 
    }
  } else if (theMuonType==2) { // StandaloneMuons

    // Standalone muons:
    // -----------------
    Handle<reco::TrackCollection> saMuons;
    event.getByLabel(theMuonLabel, saMuons);
    if (debug)
      cout << "Standalone muons: " << saMuons->size() << endl;

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
    Handle<reco::TrackCollection> tracks;
    event.getByLabel(theMuonLabel, tracks);
    if (debug)
      cout << "Tracker tracks: " << tracks->size() << endl;

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
    cout << "Wrong muon type! Aborting." << endl;
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
        cout << "  Reconstructed muon: pT = " << muon1->p4().Pt()
             << "  Eta = " << muon1->p4().Eta() << endl;
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
            vector<double>::const_iterator mMinCut = Mmin.begin();
            vector<double>::const_iterator mMaxCut = Mmax.begin();
            for( ; mMinCut != Mmin.end(); ++mMinCut, ++mMaxCut ) {
              // When the two borders are -1 do not cut.
              if( *mMinCut == *mMaxCut && *mMaxCut == -1) {
                resfound = true;
                if (debug) {
                  cout << "Acceptiong event because mMinCut = " << *mMinCut << " = mMaxCut = " << *mMaxCut << endl;
                }
              }
              else if (Z.mass()>*mMinCut && Z.mass()<*mMaxCut) {
                resfound = true;
                if (debug) {
                  cout << "One particle found with mass = " << Z.mass() << endl;
                }
              }
            }
          }
        }
      } else if (debug) {
        cout << "Not enough reconstructed muons to make a resonance" << endl; 
      }
    }
  }
  else if (debug) {
    cout << "Skipping event because muons = " << muons->size() << " < " << "minimumMuonsNumber("<<minimumMuonsNumber<<")" << endl;
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
