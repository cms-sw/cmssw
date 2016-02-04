#ifndef RecoJets_JetAssociationProducers_interface_TrackExtrapolator_h
#define RecoJets_JetAssociationProducers_interface_TrackExtrapolator_h

// -*- C++ -*-
//
// Package:    TrackExtrapolator
// Class:      TrackExtrapolator
// 
/**\class TrackExtrapolator TrackExtrapolator.cc RecoTracker/TrackExtrapolator/src/TrackExtrapolator.cc

 Description: Extrapolates tracks to Calo Face. Migrating this functionality from 
              RecoJets/JetAssociationAlgorithms/JetTracksAssociatorDRCalo.h,
	      which will now essentially be defunct. 

 Implementation:

*/
//
// Original Author:  Salvatore Rappoccio (salvatore.rappoccio@cern.ch)
//         Created:  Mon Feb 22 11:54:41 CET 2010
// $Id: TrackExtrapolator.h,v 1.3 2011/02/16 17:02:13 stadie Exp $
//
// Revision by: John Paul Chou (chou@hep.brown.edu)
//              Modified algorithm to extrapolate correctly to the endcap front face.
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/JetReco/interface/TrackExtrapolation.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrackAssociator/interface/FiducialVolume.h"

//
// class declaration
//

class TrackExtrapolator : public edm::EDProducer {
   public:
      explicit TrackExtrapolator(const edm::ParameterSet&);
      ~TrackExtrapolator();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------

      edm::InputTag tracksSrc_;    /// Input tracks
      reco::TrackBase::TrackQuality trackQuality_; /// track quality of the tracks we care about


      // ----------internal functions ---------------------------

      /// Propagate a track to a given radius, given the magnetic
      /// field and the propagator. Store the resulting
      /// position, momentum, and direction. 
      bool propagateTrackToVolume( const reco::Track& fTrack,
				   const MagneticField& fField,
				   const Propagator& fPropagator,
				   const FiducialVolume& volume,
				   reco::TrackBase::Point & resultPos,
				   reco::TrackBase::Vector & resultMom
				   );

      

};


#endif
