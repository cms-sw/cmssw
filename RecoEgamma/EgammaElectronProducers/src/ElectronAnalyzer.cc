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
// $Id$
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
#include "DataFormats/EgammaCandidates/interface/ElectronCandidate.h"

// for Si hits
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPosCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPos.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
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
   numHits_ = new TH1F("numHits", "Number of Si tracker hits associated with each electron", 20, -0.5, 19.5);
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
   iEvent.getByLabel("VtxSmeared", "", mctruthHandle);
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
   iEvent.getByLabel("superclusterproducer", "superclusterCollection", clusterHandle);
   
   for (reco::SuperClusterCollection::const_iterator clusterIter = clusterHandle->begin();
	clusterIter != clusterHandle->end();
	++clusterIter) {
      double energy = clusterIter->energy();
      math::XYZPoint position = clusterIter->position();

      cout << "supercluster " << energy << " GeV, position " << position << " cm" << endl;
   }

   // DataFormats/HLTReco/interface/HLTElectron.h
   edm::Handle<reco::ElectronCandidateCollection> electronHandle;
   iEvent.getByLabel("electronproducer", "SiStripElectronCandidateCollection", electronHandle);

   for (reco::ElectronCandidateCollection::const_iterator electronIter = electronHandle->begin();
	electronIter != electronHandle->end();
	++electronIter) {
      cout << "electron charge " << electronIter->charge() << ", momentum " << electronIter->p4() << endl;
      numHits_->Fill(5);
   }
}

//
// const member functions
//

//
// static member functions
//
