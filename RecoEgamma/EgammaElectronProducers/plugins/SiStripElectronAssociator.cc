// -*- C++ -*-
//
// Package:    SiStripElectronAssociator
// Class:      SiStripElectronAssociator
// 
/**\class SiStripElectronAssociator SiStripElectronAssociator.cc RecoEgamma/SiStripElectronAssociator/src/SiStripElectronAssociator.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Pivarski
//         Created:  Tue Aug  1 15:24:02 EDT 2006
// $Id: SiStripElectronAssociator.cc,v 1.6 2010/03/06 12:45:47 sani Exp $
//
//


#include <map>
#include <sstream>

#include "SiStripElectronAssociator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/EgammaCandidates/interface/SiStripElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/SiStripElectron.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
SiStripElectronAssociator::SiStripElectronAssociator(const edm::ParameterSet& iConfig)
{
   //register your products
  electronsLabel_ = iConfig.getParameter<edm::InputTag>("electronsLabel");
  produces<reco::ElectronCollection>(electronsLabel_.label());

   //now do what ever other initialization is needed
  //siStripElectronProducer_ = iConfig.getParameter<edm::InputTag>("siStripElectronProducer");
  siStripElectronCollection_ = iConfig.getParameter<edm::InputTag>("siStripElectronCollection");
  //trackProducer_ = iConfig.getParameter<edm::InputTag>("trackProducer");
  trackCollection_ = iConfig.getParameter<edm::InputTag>("trackCollection");
}


SiStripElectronAssociator::~SiStripElectronAssociator()
{}


void
SiStripElectronAssociator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // position tolerance for equality of 2 hits set to 10 microns
  static const double positionTol = 1e-3 ; 

   edm::Handle<reco::SiStripElectronCollection> siStripElectrons;
   iEvent.getByLabel(siStripElectronCollection_, siStripElectrons);

   edm::Handle<reco::TrackCollection> tracks;
   iEvent.getByLabel(trackCollection_, tracks);

   std::map<const reco::SiStripElectron*, bool> alreadySeen;
   for (reco::SiStripElectronCollection::const_iterator strippyIter = siStripElectrons->begin();  strippyIter != siStripElectrons->end();  ++strippyIter) {
      alreadySeen[&(*strippyIter)] = false;
   }

   // Output the high-level Electrons
   std::auto_ptr<reco::ElectronCollection> output(new reco::ElectronCollection);

   LogDebug("SiStripElectronAssociator") << " About to loop over tracks " << std::endl ;
   LogDebug("SiStripElectronAssociator") << " Number of tracks in Associator " << tracks.product()->size() ;
   LogDebug("SiStripElectronAssociator") << " Number of SiStripElectron Candidates " << siStripElectrons->size();

   // The reco::Track's hits are a (improper?) subset of the reco::SiStripElectron's
   // countSiElFit counts electron candidates that return w/ a good track fit
   int countSiElFit = 0 ;
   for (unsigned int i = 0;  i < tracks.product()->size();  i++) {
      const reco::Track* trackPtr = &(*reco::TrackRef(tracks, i));
      
      // If the reco::Track and the reco::SiStripElectron share even
      // one hit in common, they belong to each other.  (Disjoint sets
      // of hits are assigned to electrons.)  So let's look at one hit.

      // But first, make sure the track's hit list is not empty.
      if (trackPtr->recHitsBegin() == trackPtr->recHitsEnd()) { continue; }

      // Detector id is not enough to completely specify a hit
      uint32_t id = (*trackPtr->recHitsBegin())->geographicalId().rawId();
      LocalPoint pos = (*trackPtr->recHitsBegin())->localPosition();
      
      LogDebug("SiStripElectronAssociator") << " New Track Candidate " << i
                                            << " DetId " << id
                                            << " pos " << pos << "\n";

      // Find the electron with that hit!
      bool foundElectron = false;
      for (reco::SiStripElectronCollection::const_iterator strippyIter = siStripElectrons->begin();  strippyIter != siStripElectrons->end();  ++strippyIter) {
        if (!alreadySeen[&(*strippyIter)]) {
          
          bool hitInCommon = false;
          LogDebug("SiStripElectronAssociator") << " Looping over Mono hits " << "\n" ;
          
          for (std::vector<SiStripRecHit2D>::const_iterator hitIter = strippyIter->rphiRecHits().begin();  hitIter != strippyIter->rphiRecHits().end();  ++hitIter) {
            
            LogDebug("SiStripElectronAssociator") << " SiStripCand " 
                                                  << " DetId " << hitIter->geographicalId().rawId()
                                                  << " localPos " << hitIter->localPosition()
                                                  << " deltasPos " << (hitIter->localPosition() - pos).mag() ;
            
            if (hitIter->geographicalId().rawId() == id   &&
                (hitIter->localPosition() - pos).mag() < positionTol ) {
              hitInCommon = true;
              LogDebug("SiStripElectronAssociator") << " hitInCommon True " << "\n" ;
              break;
            } else {
              LogDebug("SiStripElectronAssociator") << " hitInCommon False " << "\n" ;
            }
          } // end loop over rphi hits
          
          if(!hitInCommon) {
            LogDebug("SiStripElectronAssociator") << " Looping over Stereo hits " << "\n" ;

            for (std::vector<SiStripRecHit2D>::const_iterator hitIter = strippyIter->stereoRecHits().begin();  hitIter != strippyIter->stereoRecHits().end();  ++hitIter) {
              
               LogDebug("SiStripElectronAssociator") << " SiStripCand " 
                                                     << " DetId " << hitIter->geographicalId().rawId()
                                                     << " localPos " << hitIter->localPosition()
                                                     << " deltasPos " << (hitIter->localPosition() - pos).mag() ;
               
               if (hitIter->geographicalId().rawId() == id   &&
                   (hitIter->localPosition() - pos).mag() < positionTol) {
                 hitInCommon = true;
                  LogDebug("SiStripElectronAssociator") << " hitInCommon True " << "\n"  ;
                 break;
               } else {
                 LogDebug("SiStripElectronAssociator") << " hitInCommon False " << "\n" ;
               }
               
            } // end loop over stereo hits
          } // end of hitInCommon check for loop over stereo hits
          
          if (hitInCommon) {
             LogDebug("SiStripElectronAssociator") << " Hit in Common Found \n" ;
             ++countSiElFit ;
             foundElectron = true;
             alreadySeen[&(*strippyIter)] = true;
             
             reco::Electron electron((trackPtr->charge() > 0 ? 1 : -1),
                                     math::XYZTLorentzVector(trackPtr->px(),
                                                             trackPtr->py(),
                                                             trackPtr->pz(),
                                                             trackPtr->p()),
                                     math::XYZPoint(trackPtr->vx(),
                                                    trackPtr->vy(),
                                                    trackPtr->vz()));
             electron.setSuperCluster(strippyIter->superCluster());
             electron.setTrack(reco::TrackRef(tracks, i));
             
             output->push_back(electron);
             break ; // no need to check other SiStripElectrons if this one is good 
             
          } else {
             LogDebug("SiStripElectronAssociator") << "Hit in Common NOT found \n"  ;
          }
          // endif this electron belongs to this track
          
          
        } // endif we haven't seen this electron before
        LogDebug("SiStripElectronAssociator") << "Done with this electron " << "\n\n\n";
      } // end loop over electrons
      
      LogDebug("SiStripElectronAssociator") << "Testing if foundElectron " << foundElectron << std::endl;
      
      if (!foundElectron) {
        throw cms::Exception("Configuration")
          << " It is possible that the trackcollection used '"
          << trackCollection_ << "' from producer '" << trackProducer_
          << "' is not consistent with '"<< siStripElectronCollection_ 
          << "' from the producer '"<< siStripElectronProducer_
          << "' --- Please check your cfg file " << "\n";
      }
      
      LogDebug("SiStripElectronAssociator") << "At end of track loop \n" << std::endl; 
      
   } // end loop over tracks
   
   
   LogDebug("SiStripElectronAssociator") << " Number of SiStripElectrons returned with a good fit " 
                                         << countSiElFit << "\n"<<  std::endl ;
   iEvent.put(output, electronsLabel_.label());
}
