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
// $Id: ElectronAnalyzer.cc,v 1.1 2006/05/27 04:29:29 pivarski Exp $
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
#include "DataFormats/EgammaCandidates/interface/SiStripElectronCandidate.h"

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
   numCand_ = new TH1F("numCandidates", "Number of candidates found", 10, -0.5, 9.5);
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

   // DataFormats/EgammaCandidates/src/SiStripElectronCandidate.cc
   edm::Handle<reco::SiStripElectronCandidateCollection> electronHandle;
   iEvent.getByLabel("electronProd", "SiStripElectronCandidate", electronHandle);

   int numberOfElectrons = 0;
   for (reco::SiStripElectronCandidateCollection::const_iterator electronIter = electronHandle->begin();
	electronIter != electronHandle->end();
	++electronIter) {
      numberOfElectrons++;
   }
   numCand_->Fill(numberOfElectrons);
}

//
// const member functions
//

//
// static member functions
//
