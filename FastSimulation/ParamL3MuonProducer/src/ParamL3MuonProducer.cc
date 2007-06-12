//
// Package:    ParamL3MuonProducer
// Class:      ParamL3MuonProducer
// 
/**\class ParamL3MuonProducer FastSimulation/ParamL3MuonProducer/src/ParamL3MuonProducer.cc

 Description:
    Fast simulation producer for L1, L3 and Global muons.
    L1MuGMTCand's obtained from a parameterization wich starts from the generated
              muons in the event
    L3 muons obtained from a parameterization wich starts from the L1 muon seed and
              from the corresponding reconstructed track in the tracker.
    Global muons obtained from a parameterization wich link to the L3 and/or the
              corresponding reconstructed track in the tracker.

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Perrotta
//         Created:  Mon Oct 30 14:37:24 CET 2006
// $Id: ParamL3MuonProducer.cc,v 1.2 2007/06/11 14:51:23 aperrott Exp $
//
//

// CMSSW headers 
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/PluginManager.h"

// Fast Simulation headers
#include "FastSimulation/Utilities/interface/RandomEngine.h"
#include "FastSimulation/ParamL3MuonProducer/interface/ParamL3MuonProducer.h"

// SimTrack
#include "SimDataFormats/Track/interface/SimTrack.h"

// L1
#include "FastSimulation/ParamL3MuonProducer/interface/SimpleL1MuGMTCand.h"
#include "FastSimulation/ParamL3MuonProducer/interface/FML1EfficiencyHandler.h"
#include "FastSimulation/ParamL3MuonProducer/interface/FML1PtSmearer.h"

// L3
#include "FastSimulation/ParamL3MuonProducer/interface/FML3EfficiencyHandler.h"
#include "FastSimulation/ParamL3MuonProducer/interface/FML3PtSmearer.h"

// GL
#include "FastSimulation/ParamL3MuonProducer/interface/FMGLfromL3EfficiencyHandler.h"
#include "FastSimulation/ParamL3MuonProducer/interface/FMGLfromL3TKEfficiencyHandler.h"
#include "FastSimulation/ParamL3MuonProducer/interface/FMGLfromTKEfficiencyHandler.h"

// STL headers 
#include <vector>
#include <iostream>

// CLHEP headers
#include "DataFormats/Math/interface/LorentzVector.h"

// Data Formats
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2DCollection.h"
#include "DataFormats/MuonReco/interface/Muon.h"


// constants, enums and typedefs
typedef std::vector<L1MuGMTCand> L1MuonCollection;


//
// static data member definitions
//

//
// constructors and destructor
//
ParamL3MuonProducer::ParamL3MuonProducer(const edm::ParameterSet& iConfig)
{

  readParameters(iConfig.getParameter<edm::ParameterSet>("MUONS"));

  //register your products
  if (doL1_) produces<std::vector<L1MuGMTCand> >("ParamL1Muons");
  if (doL3_) produces<reco::MuonCollection>("ParamL3Muons");
  if (doGL_) produces<reco::MuonCollection>("ParamGlobalMuons");

  // Initialize the random number generator service
  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable() ) {
    throw cms::Exception("Configuration") <<
      "ParamMuonProducer requires the RandomGeneratorService \n"
      "which is not present in the configuration file. \n"
      "You must add the service in the configuration file\n"
      "or remove the module that requires it.";
  }
  //  CLHEP::HepRandomEngine& random = rng->getEngine();
  random = new RandomEngine(&(*rng));

}


ParamL3MuonProducer::~ParamL3MuonProducer()
{
 
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
  
  if ( random ) delete random;
}


//
// member functions
//

// ------------ method called to produce the data  ------------

void ParamL3MuonProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  Handle<std::vector<SimTrack> > simMuons;
  iEvent.getByLabel(theSimModuleLabel_,"MuonSimTracks",simMuons);
  //  Handle<std::vector<SimVertex> > simVertices;
  //  iEvent.getByLabel(theSimModuleLabel_,simVertices);

  int ntrks = 0;
  Handle<reco::TrackCollection> theTracks;
  reco::TrackCollection::const_iterator trk;
  reco::TrackRefVector allMuonTracks;
  Handle<SiTrackerGSRecHit2DCollection> theGSRecHits;
  std::vector<unsigned int> SimTrackIds;
  std::vector<SimTrack> trackOriginalMuons;  

  if (doL3_ || doGL_) {
    iEvent.getByLabel(theTrkModuleLabel_,theTracks);
    ntrks =  theTracks->size();
    //Get RecHits from the event
    iEvent.getByType(theGSRecHits);

// -- Associate the reconstructed trackerTrack with the simTrack...
    int trackIndex = 0;
    for (trk=theTracks->begin(); trk!=theTracks->end(); trk++) {
      SimTrackIds.clear();
      for (trackingRecHit_iterator it=trk->recHitsBegin(); it!=trk->recHitsEnd(); it++) {
	if ((*it)->isValid()) {
	  if(const SiTrackerGSRecHit2D * rechit = dynamic_cast<const SiTrackerGSRecHit2D *> (it->get())) {
	    int currentId = rechit->simtrackId();
	    SimTrackIds.push_back(currentId);
	  }
	}
      }  // end of loop over the recHits belonging to the track

      int nmax = 0;
      int idmax = -1;
      for(size_t j=0; j<SimTrackIds.size(); j++){
	int n = std::count(SimTrackIds.begin(), SimTrackIds.end(), SimTrackIds[j]);
	if(n>nmax){
	  nmax = n;
	  idmax = SimTrackIds[j];
	}
      }

      for( unsigned fsimi=0; fsimi < simMuons->size(); ++fsimi) {
	const SimTrack& simTrack = (*simMuons)[fsimi];
	if( (int) simTrack.trackId() == idmax) {
	  allMuonTracks.push_back(reco::TrackRef(theTracks,trackIndex));
	  trackOriginalMuons.push_back(simTrack);
	  break;
	}
      }

      trackIndex++;
    } // end loop on rec tracks
  }   // end if clause


  if (debug_) {
    std::cout << " *** ParamMuonProducer::reconstruct() -> entering " << std::endl;
    std::cout << " *** Event with " << ntrks << " tracker tracks" << std::endl;
  }

//
// Loop over generated muons and reconstruct L1, L3 and Global muons
//
  
  int nMu = 0;
  mySimpleL1MuonCands.clear();
  mySimpleL3MuonCands.clear();
  mySimpleGLMuonCands.clear();

  for( unsigned fsimi=0; fsimi < simMuons->size(); ++fsimi) {
    const SimTrack& mySimTrack = (*simMuons)[fsimi];

    bool hasL1 = false , hasL3 = false , hasTK = false , hasGL = false;
    //Replace with this as soon transition to ROOTMath is complete
    //    math::XYZTLorentzVector& mySimP4 =  mySimTrack.momentum();

    math::XYZTLorentzVector mySimP4 =  math::XYZTLorentzVector(mySimTrack.momentum().x(),
							       mySimTrack.momentum().y(),
							       mySimTrack.momentum().z(),
							       mySimTrack.momentum().t());

// *** Reconstruct parameterized muons starting from undecayed simulated muons
 
    if ( mySimP4.Eta()>minEta_ && mySimP4.Eta()<maxEta_ ) {
      
      nMu++;
      if (debug_) {
	std::cout << " ===> ParamMuonProducer::reconstruct() - pid = "
		  << mySimTrack.type() ;
	std::cout << " : pT = " << mySimP4.Pt()
		  << ", eta = " << mySimP4.Eta()
		  << ", phi = " << mySimP4.Phi() << std::endl;
      }

//
// Now L1
//

      SimpleL1MuGMTCand * thisL1MuonCand = new SimpleL1MuGMTCand(&mySimTrack);
      if (doL1_ || doL3_ || doGL_) {
	hasL1 = myL1EfficiencyHandler->kill(thisL1MuonCand);
	if (hasL1) {
	  bool status2 = myL1PtSmearer->smear(thisL1MuonCand);
	  if (!status2) { std::cout << "Pt smearing of L1 muon went wrong!!" << std::endl; }
	  if (status2 && mySimpleL1MuonCands.size()<4) {
	    mySimpleL1MuonCands.push_back(thisL1MuonCand);
	  }
	  else {
	    hasL1 = false;
	    delete thisL1MuonCand;
	  }
	}
      }


      reco::TrackRef myTrackerTrack;
      if (doL3_ || doGL_) {

// Check if a correspondig track does exist:
//	std::vector<FSimTrack>::const_iterator genmu;
	std::vector<SimTrack>::const_iterator genmu;
	reco::track_iterator trkmu=allMuonTracks.begin();
	for (genmu=trackOriginalMuons.begin();
	     genmu!=trackOriginalMuons.end();genmu++) {
	  if(mySimTrack.trackId() == (*genmu).trackId()) {
	    hasTK = true;
	    myTrackerTrack = (*trkmu);
	    break;
	  }
	  trkmu++;
	}

//
// L3 muon
//
	if (hasL1 && hasTK) {
	  hasL3 = myL3EfficiencyHandler->kill(mySimTrack);
	  if (hasL3) {
	    int myL3Charge = myTrackerTrack->charge();
	    const math::XYZTLorentzVector& myL3P4 =
	              myL3PtSmearer->smear(mySimP4,myTrackerTrack->momentum());
	    math::XYZPoint myL3Vertex = myTrackerTrack->referencePoint();
	    reco::Muon * thisL3MuonCand = new reco::Muon(myL3Charge,myL3P4,myL3Vertex);
	    thisL3MuonCand->setTrack(myTrackerTrack);
	    mySimpleL3MuonCands.push_back((*thisL3MuonCand));
	  }
	}

//
// Global Muon
//
	if (doGL_ && hasL3 && hasTK) {
	  hasGL = myGLfromL3TKEfficiencyHandler->kill(mySimTrack);
	}
	else if (doGL_ && hasTK) {
	  hasGL = myGLfromTKEfficiencyHandler->kill(mySimTrack);
	}
	//      else if (doGL_ && hasL3) {
	//	hasGL = myGLfromL3EfficiencyHandler->kill(mySimTrack);
	//      }
	if (hasGL) {
	  int myGLCharge = myTrackerTrack->charge();
	  const math::XYZTLorentzVector& myGLP4 =
          myGLPtSmearer->smear(mySimP4,myTrackerTrack->momentum());
	  math::XYZPoint myGLVertex = myTrackerTrack->referencePoint();
	  reco::Muon * thisGLMuonCand = new reco::Muon(myGLCharge,myGLP4,myGLVertex);
	  thisGLMuonCand->setTrack(myTrackerTrack);
	  mySimpleGLMuonCands.push_back((*thisGLMuonCand));
	}
      }

//
// Summary debug for this generated muon:
//
      if (debug_) {
	std::cout << " Muon " << nMu << " reconstructed with: " ;
	if (hasL1) std::cout << " L1 ; " ;
	if (hasTK) std::cout << " Tk ; " ;
	if (hasL3) std::cout << " L3 ; " ;
	if (hasGL) std::cout << " GL . " ;
	std::cout << std::endl;
      }
      
    }

  }

  int nL1 =  mySimpleL1MuonCands.size();
  int nL3 =  mySimpleL3MuonCands.size();
  int nGL =  mySimpleGLMuonCands.size();
  nMuonTot   += nMu;
  nL1MuonTot += nL1;
  nL3MuonTot += nL3;
  nGLMuonTot += nGL;

// start debug -->
  if (debug_) {
    std::cout << " ===> Number of generator -> L1 / L3 / Global muons in the event : "
              << nMu << " -> " << nL1 <<  " / " << nL3 <<  " / " << nGL << std::endl;
    unsigned int i = 0;
    FML1Muons::const_iterator l1mu;
    for (l1mu=mySimpleL1MuonCands.begin(); l1mu!=mySimpleL1MuonCands.end(); l1mu++) {
      ++i;
      std::cout << "FastMuon L1 Cand " << i 
		<< " : pT = " << (*l1mu)->ptValue()
		<< ", eta = " << (*l1mu)->etaValue()
		<< ", phi = " << (*l1mu)->phiValue() << std::endl;
    }
    i=0;
    reco::MuonCollection::const_iterator l3mu;
    for (l3mu=mySimpleL3MuonCands.begin(); l3mu!=mySimpleL3MuonCands.end(); l3mu++) {
      ++i;
      std::cout << "FastMuon L3 Cand " << i 
		<< " : pT = " << (*l3mu).pt()
		<< ", eta = " << (*l3mu).eta()
		<< ", phi = " << (*l3mu).phi() 
	//                << ", vertex = ( " << (*l3mu).vx()
	//                                   << " , " << (*l3mu).vy()
	//		                     << " , " << (*l3mu).vz() << " )"
		<< std::endl;
      std::cout << "-    tracker Track" 
		<< " : pT = " << (*l3mu).track()->pt()
		<< ", eta = " << (*l3mu).track()->eta()
		<< ", phi = " << (*l3mu).track()->phi()
	//                << ", vertex = ( " << (*l3mu).track()->vx()
	//                                   << " , " << (*l3mu).track()->vy()
	//		                     << " , " << (*l3mu).track()->vz() << " )"
		<< std::endl;
    }
    i=0;
    reco::MuonCollection::const_iterator glmu;
    for (glmu=mySimpleL3MuonCands.begin(); glmu!=mySimpleL3MuonCands.end(); glmu++) {
      ++i;
      std::cout << "FastGlobalMuon Cand " << i 
		<< " : pT = " << (*glmu).pt()
		<< ", eta = " << (*glmu).eta()
		<< ", phi = " << (*glmu).phi() 
	//                << ", vertex = ( " << (*l3mu).vx()
	//                                   << " , " << (*l3mu).vy()
	//		                     << " , " << (*l3mu).vz() << " )"
		<< std::endl;
      std::cout << "-    tracker Track" 
		<< " : pT = " << (*glmu).track()->pt()
		<< ", eta = " << (*glmu).track()->eta()
		<< ", phi = " << (*glmu).track()->phi()
	//                << ", vertex = ( " << (*l3mu).track()->vx()
	//                                   << " , " << (*l3mu).track()->vy()
	//		                     << " , " << (*l3mu).track()->vz() << " )"
		<< std::endl;
    }
    
  }
// end debug -->

  if (doL1_) {
    std::auto_ptr<std::vector<L1MuGMTCand> > l1Out(new std::vector<L1MuGMTCand>);
    loadL1Muons(*l1Out);
    iEvent.put(l1Out,"ParamL1Muons");
  }
  if (doL3_) {
    std::auto_ptr<reco::MuonCollection> l3Out(new reco::MuonCollection);
    loadL3Muons(*l3Out);
    iEvent.put(l3Out,"ParamL3Muons");
  }
  if (doGL_) {
    std::auto_ptr<reco::MuonCollection> glOut(new reco::MuonCollection);
    loadGLMuons(*glOut);
    iEvent.put(glOut,"ParamGlobalMuons");
  }

}


void ParamL3MuonProducer::loadL1Muons(L1MuonsContainer & c) const
{
  FML1Muons::const_iterator l1mu;
  // Add L1 muons:
  for(l1mu=mySimpleL1MuonCands.begin();l1mu!=mySimpleL1MuonCands.end();++l1mu) {
    c.push_back(*(*l1mu));
  }
}

void ParamL3MuonProducer::loadL3Muons(reco::MuonCollection & c) const
{
  reco::MuonCollection::const_iterator l3mu;
  // Add L3 muons:
  for(l3mu=mySimpleL3MuonCands.begin();l3mu!=mySimpleL3MuonCands.end();++l3mu) {
    c.push_back(*l3mu);
  }
}

void ParamL3MuonProducer::loadGLMuons(reco::MuonCollection & c) const
{
  reco::MuonCollection::const_iterator glmu;
  // Add Global muons:
  for(glmu=mySimpleGLMuonCands.begin();glmu!=mySimpleGLMuonCands.end();++glmu) {
    c.push_back(*glmu);
  }
}


// ------------ method called once each job just before starting event loop  ------------
void ParamL3MuonProducer::beginJob(const edm::EventSetup& es)
{

  // Initialize
  nMuonTot = 0;

  nL1MuonTot = 0;
  mySimpleL1MuonCands.clear();
  myL1EfficiencyHandler = new FML1EfficiencyHandler(random);
  myL1PtSmearer = new FML1PtSmearer(random);

  nL3MuonTot = 0;
  mySimpleL3MuonCands.clear();
  myL3EfficiencyHandler = new FML3EfficiencyHandler(random);
  myL3PtSmearer = new FML3PtSmearer(random);

  nGLMuonTot = 0;
  mySimpleGLMuonCands.clear();
  myGLfromL3TKEfficiencyHandler = new FMGLfromL3TKEfficiencyHandler(random);
  myGLfromL3EfficiencyHandler = new FMGLfromL3EfficiencyHandler(random);
  myGLfromTKEfficiencyHandler = new FMGLfromTKEfficiencyHandler(random);
  //  myGLPtSmearer = new FML3PtSmearer(random);
  myGLPtSmearer = myL3PtSmearer;

}


// ------------ method called once each job just after ending the event loop  ------------
void ParamL3MuonProducer::endJob() {

  std::cout << " ===> ParamL3MuonProducer , final report." << std::endl;
  std::cout << " ===> Number of total -> L1 / L3 / GL muons in the whole run : "
            <<   nMuonTot << " -> " << nL1MuonTot << " / "
	    << nL3MuonTot << " / " << nGLMuonTot << std::endl;
}


void ParamL3MuonProducer::readParameters(const edm::ParameterSet& fastMuons) {
  debug_ = fastMuons.getUntrackedParameter<bool>("Debug");
  doL1_ = fastMuons.getUntrackedParameter<bool>("ProduceL1Muons");
  doL3_ = fastMuons.getUntrackedParameter<bool>("ProduceL3Muons");
  doGL_ = fastMuons.getUntrackedParameter<bool>("ProduceGlobalMuons");
  theSimModuleLabel_ = fastMuons.getParameter<std::string>("simModuleLabel");
  theTrkModuleLabel_ = fastMuons.getParameter<std::string>("trackModuleLabel");
  minEta_ = fastMuons.getParameter<double>("MinEta");
  maxEta_ = fastMuons.getParameter<double>("MaxEta");
  if (minEta_ > maxEta_) {
    double tempEta_ = maxEta_ ;
    maxEta_ = minEta_ ;
    minEta_ = tempEta_ ;
  }

  std::cout << " Parameterized MUONS: FastSimulation parameters " << std::endl;
  std::cout << " ============================================== " << std::endl;
  std::cout << " Parameterized muons reconstructed in the pseudorapidity range : "
            << minEta_ << " -> " << maxEta_ << std::endl;
}


//define this as a plug-in
DEFINE_FWK_MODULE(ParamL3MuonProducer);
