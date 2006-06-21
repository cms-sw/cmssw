// -*- C++ -*-
//
// Package:     EgammaElectronProducers
// Class  :     ElectronAnalyzer
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Jim Pivarski
//         Created:  Fri May 26 16:49:38 EDT 2006
// $Id: ElectronAnalyzer.cc,v 1.2 2006/06/21 17:02:05 pivarski Exp $
//

// system include files
#include <memory>

// user include files
#include "RecoEgamma/EgammaElectronProducers/interface/ElectronAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Math/interface/Point3D.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaCandidates/interface/SiStripElectron.h"

// for Si hits
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPosCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPos.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Vector/interface/GlobalPoint.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
ElectronAnalyzer::ElectronAnalyzer(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
   fileName_ = iConfig.getParameter<std::string>("fileName");

   file_ = new TFile(fileName_.c_str(), "RECREATE");
   numCand_ = new TH1F("numCandidates", "Number of candidates found", 10, -0.5, 9.5);

   mctruthProducer_ = iConfig.getParameter<std::string>("mctruthProducer");
   mctruthCollection_ = iConfig.getParameter<std::string>("mctruthCollection");

   superClusterProducer_ = iConfig.getParameter<std::string>("superClusterProducer");
   superClusterCollection_ = iConfig.getParameter<std::string>("superClusterCollection");

   electronProducer_ = iConfig.getParameter<std::string>("electronProducer");
   electronCollection_ = iConfig.getParameter<std::string>("electronCollection");

   siHitProducer_ = iConfig.getParameter<std::string>("siHitProducer");
   siRphiHitCollection_ = iConfig.getParameter<std::string>("siRphiHitCollection");
   siStereoHitCollection_ = iConfig.getParameter<std::string>("siStereoHitCollection");
}

// ElectronAnalyzer::ElectronAnalyzer(const ElectronAnalyzer& rhs)
// {
//    // do actual copying here;
// }

ElectronAnalyzer::~ElectronAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

   file_->Write();
   file_->Close();
}

//
// assignment operators
//
// const ElectronAnalyzer& ElectronAnalyzer::operator=(const ElectronAnalyzer& rhs)
// {
//   //An exception safe implementation is
//   ElectronAnalyzer temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

// ------------ method called to produce the data  ------------
void
ElectronAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace std;  // so you can say "cout" and "endl"
   
   // http://cmsdoc.cern.ch/swdev/lxr/CMSSW/source/clhep/CLHEP/HepMC/GenParticle.h
   // http://cmsdoc.cern.ch/swdev/lxr/CMSSW/source/clhep/CLHEP/HepMC/GenVertex.h
   edm::Handle<edm::HepMCProduct> mctruthHandle;
   iEvent.getByLabel(mctruthProducer_, mctruthCollection_, mctruthHandle);
   HepMC::GenEvent mctruth = mctruthHandle->getHepMCData();
   
   for (HepMC::GenEvent::particle_const_iterator partIter = mctruth.particles_begin();
	partIter != mctruth.particles_end();
	++partIter) {
//    for (HepMC::GenEvent::vertex_const_iterator vertIter = mctruth.vertices_begin();
// 	vertIter != mctruth.vertices_end();
// 	++vertIter) {
      CLHEP::HepLorentzVector creation = (*partIter)->CreationVertex();
      CLHEP::HepLorentzVector momentum = (*partIter)->Momentum();
      HepPDT::ParticleID id = (*partIter)->particleID();  // electrons and positrons are 11 and -11
      cout << "MC particle id " << id.pid() << ", creationVertex " << creation << " cm, initialMomentum " << momentum << " GeV/c" << endl;
   }
   
   // http://cmsdoc.cern.ch/swdev/lxr/CMSSW/source/self/DataFormats/EgammaReco/interface/SuperCluster.h
   edm::Handle<reco::SuperClusterCollection> clusterHandle;
   iEvent.getByLabel(superClusterProducer_, superClusterCollection_, clusterHandle);
   
   for (reco::SuperClusterCollection::const_iterator clusterIter = clusterHandle->begin();
	clusterIter != clusterHandle->end();
	++clusterIter) {
      double energy = clusterIter->energy();
      math::XYZPoint position = clusterIter->position();

      cout << "supercluster " << energy << " GeV, position " << position << " cm" << endl;
   }

   // DataFormats/EgammaCandidates/src/SiStripElectron.cc
   edm::Handle<reco::SiStripElectronCollection> electronHandle;
   iEvent.getByLabel(electronProducer_, electronCollection_, electronHandle);

   int numberOfElectrons = 0;
   for (reco::SiStripElectronCollection::const_iterator electronIter = electronHandle->begin();
	electronIter != electronHandle->end();
	++electronIter) {
      cout << "about to get stuff from electroncandidate..." << endl;
      cout << "supercluster energy = " << electronIter->superCluster()->energy() << endl;
      cout << "fit results are phi(r) = " << electronIter->phiAtOrigin() << " + " << electronIter->phiVsRSlope() << "*r" << endl;
      cout << "you get the idea..." << endl;

      numberOfElectrons++;
   }
   numCand_->Fill(numberOfElectrons);

   ///////////////////////////////////////////////////////////////////////////////// Now for tracker hits:
   edm::ESHandle<TrackerGeometry> trackerHandle;
   iSetup.get<TrackerDigiGeometryRecord>().get(trackerHandle);

   edm::Handle<SiStripRecHit2DLocalPosCollection> rphiHitsHandle;
   iEvent.getByLabel(siHitProducer_, siRphiHitCollection_, rphiHitsHandle);

   edm::Handle<SiStripRecHit2DLocalPosCollection> stereoHitsHandle;
   iEvent.getByLabel(siHitProducer_, siStereoHitCollection_, stereoHitsHandle);

   // Loop over the detector ids
   const std::vector<DetId> ids = stereoHitsHandle->ids();
   for (std::vector<DetId>::const_iterator id = ids.begin();  id != ids.end();  ++id) {

      // Get the hits on this detector id
      SiStripRecHit2DLocalPosCollection::range hits = stereoHitsHandle->get(*id);

      // Count the number of hits on this detector id
      unsigned int numberOfHits = 0;
      for (SiStripRecHit2DLocalPosCollection::const_iterator hit = hits.first;  hit != hits.second;  ++hit) {
	 numberOfHits++;
      }
      
      // Only take the hits if there aren't too many
      // (Would it be better to loop only once, fill a temporary list,
      // and copy that if numberOfHits <= maxHitsOnDetId_?)
      if (numberOfHits <= 5) {
	 for (SiStripRecHit2DLocalPosCollection::const_iterator hit = hits.first;  hit != hits.second;  ++hit) {
	    if (trackerHandle->idToDetUnit(hit->geographicalId())->type().subDetector() == GeomDetType::TIB  ||
		trackerHandle->idToDetUnit(hit->geographicalId())->type().subDetector() == GeomDetType::TOB    ) {

	       GlobalPoint position = trackerHandle->idToDet(hit->geographicalId())->surface().toGlobal(hit->localPosition());
	       cout << "this stereo hit is at " << position.x() << ", " << position.y() << ", " << position.z() << endl;

	    } // end if this is the right subdetector
	 } // end loop over hits
      } // end if this detector id doesn't have too many hits on it
   }
}

//
// const member functions
//

//
// static member functions
//
