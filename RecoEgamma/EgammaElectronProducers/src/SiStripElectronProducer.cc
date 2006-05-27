// -*- C++ -*-
//
// Package:     EgammaElectronProducers
// Class  :     SiStripElectronProducer
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Fri May 26 16:11:30 EDT 2006
// $Id$
//

// system include files
#include <memory>

// user include files
#include "RecoEgamma/EgammaElectronProducers/interface/SiStripElectronProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPosCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPos.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
SiStripElectronProducer::SiStripElectronProducer(const edm::ParameterSet& iConfig)
{
   //register your products
   produces<reco::SiStripElectronCandidateCollection>("SiStripElectronCandidateCollection");

   //now do what ever other initialization is needed
   siHitProducer_ = iConfig.getParameter<std::string>("siHitProducer");
   siRphiHitCollection_ = iConfig.getParameter<std::string>("siRphiHitCollection");
   siStereoHitCollection_ = iConfig.getParameter<std::string>("siStereoHitCollection");

   superClusterProducer_ = iConfig.getParameter<std::string>("superClusterProducer");
   superClusterCollection_ = iConfig.getParameter<std::string>("superClusterCollection");
   
   algo_p = new SiStripElectronAlgo();
}


// SiStripElectronProducer::SiStripElectronProducer(const SiStripElectronProducer& rhs)
// {
//    // do actual copying here;
// }

SiStripElectronProducer::~SiStripElectronProducer()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
   delete algo_p;
}

//
// assignment operators
//
// const SiStripElectronProducer& SiStripElectronProducer::operator=(const SiStripElectronProducer& rhs)
// {
//   //An exception safe implementation is
//   SiStripElectronProducer temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

// ------------ method called to produce the data  ------------
void
SiStripElectronProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   // Extract data from the event
   edm::ESHandle<TrackerGeometry> trackerHandle;
   iSetup.get<TrackerDigiGeometryRecord>().get(trackerHandle);

   edm::Handle<SiStripRecHit2DLocalPosCollection> rphiHitsHandle;
   iEvent.getByLabel(siHitProducer_, siRphiHitCollection_, rphiHitsHandle);
   const SiStripRecHit2DLocalPosCollection* rphiHits = rphiHitsHandle.product();
   const std::vector<DetId> rphiDetIds = rphiHits->ids();

   edm::Handle<SiStripRecHit2DLocalPosCollection> stereoHitsHandle;
   iEvent.getByLabel(siHitProducer_, siRphiHitCollection_, stereoHitsHandle);
   const SiStripRecHit2DLocalPosCollection* stereoHits = stereoHitsHandle.product();
   const std::vector<DetId> stereoDetIds = stereoHits->ids();

   edm::Handle<reco::SuperClusterCollection> superClusterHandle;
   iEvent.getByLabel(superClusterProducer_, superClusterCollection_, superClusterHandle);

   // Prepare an output electron collection
   std::auto_ptr<reco::SiStripElectronCandidateCollection> electronOut(new reco::SiStripElectronCandidateCollection);

   // Loop over clusters
//    for (reco::SuperClusterCollection::const_iterator superClusterIter = superClusterHandle->begin();  superClusterIter != superClusterHandle->end();  ++superClusterIter) {
//       double energy = superClusterIter->energy();
//       double x = superClusterIter->position().x();
//       double y = superClusterIter->position().y();
//       double z = superClusterIter->position().z();

//       // Everything passes for now
//       if (true  &&  energy*energy + x*x + y*y + z*z > -1.) {
// 	 reco::ElectronCandidate electron(-1, LorentzVector(1, 2, 3, 4));
// 	 electronOut->push_back(electron);
//       }
//    }

//   std::cout << "PAYATTENTION " << label << "_barrelrphi.append(( \\" << std::endl;
//   for (std::vector<DetId>::const_iterator id = rphiDetectorIds.begin();  id != rphiDetectorIds.end();  ++id) {
//     SiStripRecHit2DLocalPosCollection::range detHits = rphiRecHits->get(*id);
//     for (SiStripRecHit2DLocalPosCollection::const_iterator detHit = detHits.first;  detHit != detHits.second;  ++detHit) {
//       GlobalPoint threePoint = tracker->idToDet(detHit->geographicalId())->surface().toGlobal(detHit->localPosition());
//       if ((threePoint.perp2() < 55.*55.  &&  fabs(threePoint.z()) < 67.)  ||  (threePoint.perp2() >= 55.*55.  &&  fabs(threePoint.z()) < 115.)) {
// 	std::cout << "PAYATTENTION   {'r':" << sqrt(threePoint.perp2()) << ", 'phi':" << threePoint.phi() << ", 'id':" << id->rawId() << "}, \\" << std::endl;
//       } // end if barrel
//     } // end loop over hits in a detector
//   } // end loop over detectors that have been hit
//   std::cout << "PAYATTENTION   ))" << std::endl;

//   std::cout << "PAYATTENTION " << label << "_barrelstereo.append(( \\" << std::endl;
//   for (std::vector<DetId>::const_iterator id = stereoDetectorIds.begin();  id != stereoDetectorIds.end();  ++id) {
//     SiStripRecHit2DLocalPosCollection::range detHits = stereoRecHits->get(*id);
//     for (SiStripRecHit2DLocalPosCollection::const_iterator detHit = detHits.first;  detHit != detHits.second;  ++detHit) {
//       GlobalPoint threePoint = tracker->idToDet(detHit->geographicalId())->surface().toGlobal(detHit->localPosition());
//       if ((threePoint.perp2() < 55.*55.  &&  fabs(threePoint.z()) < 67.)  ||  (threePoint.perp2() >= 55.*55.  &&  fabs(threePoint.z()) < 115.)) {
// 	std::cout << "PAYATTENTION   {'r':" << sqrt(threePoint.perp2()) << ", 'phi':" << threePoint.phi() << ", 'z':" << threePoint.z() << ", 'id':" << id->rawId() << "}, \\" << std::endl;
//       } // end if barrel
//     } // end loop over hits in a detector
//   } // end loop over detectors that have been hit
//   std::cout << "PAYATTENTION   ))" << std::endl;

//   std::cout << "PAYATTENTION " << label << "_endcap.append(( \\" << std::endl;
//   for (std::vector<DetId>::const_iterator id = rphiDetectorIds.begin();  id != rphiDetectorIds.end();  ++id) {
//     SiStripRecHit2DLocalPosCollection::range detHits = rphiRecHits->get(*id);
//     for (SiStripRecHit2DLocalPosCollection::const_iterator detHit = detHits.first;  detHit != detHits.second;  ++detHit) {
//       GlobalPoint threePoint = tracker->idToDet(detHit->geographicalId())->surface().toGlobal(detHit->localPosition());
//       if ((threePoint.perp2() < 55.*55.  &&  67. < fabs(threePoint.z())  &&  fabs(threePoint.z()) < 115.)  ||  fabs(threePoint.z()) > 115.) {
// 	std::cout << "PAYATTENTION   {'phi':" << threePoint.phi() << ", 'z':" << threePoint.z() << ", 'id':" << id->rawId() << "}, \\" << std::endl;
//       } // end if barrel
//     } // end loop over hits in a detector
//   } // end loop over detectors that have been hit
//   for (std::vector<DetId>::const_iterator id = stereoDetectorIds.begin();  id != stereoDetectorIds.end();  ++id) {
//     SiStripRecHit2DLocalPosCollection::range detHits = stereoRecHits->get(*id);
//     for (SiStripRecHit2DLocalPosCollection::const_iterator detHit = detHits.first;  detHit != detHits.second;  ++detHit) {
//       GlobalPoint threePoint = tracker->idToDet(detHit->geographicalId())->surface().toGlobal(detHit->localPosition());
//       if ((threePoint.perp2() < 55.*55.  &&  67. < fabs(threePoint.z())  &&  fabs(threePoint.z()) < 115.)  ||  fabs(threePoint.z()) > 115.) {
// 	std::cout << "PAYATTENTION   {'phi':" << threePoint.phi() << ", 'z':" << threePoint.z() << ", 'id':" << id->rawId() << "}, \\" << std::endl;
//       } // end if barrel
//     } // end loop over hits in a detector
//   } // end loop over detectors that have been hit
//   std::cout << "PAYATTENTION   ))" << std::endl;

   iEvent.put(electronOut, "SiStripElectronCandidateCollection");
}

//
// const member functions
//

//
// static member functions
//
