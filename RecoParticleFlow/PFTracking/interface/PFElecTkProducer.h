#ifndef PFElecTkProducer_H
#define PFElecTkProducer_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateMode.h"
#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateTransform.h"
class PFTrackTransformer;
class GsfTrack;
class MagneticField;
class TrackerGeometry;

/// \brief Abstract
/*!
\author Michele Pioppi, Daniele Benedetti
\date January 2007

 PFElecTkProducer reads the merged GsfTracks collection
 built with the TrackerDriven and EcalDriven seeds 
 and transform them in PFGsfRecTracks.
*/

class PFElecTkProducer : public edm::EDProducer {
 public:
  
     ///Constructor
     explicit PFElecTkProducer(const edm::ParameterSet&);

     ///Destructor
     ~PFElecTkProducer();

   private:
      virtual void beginRun(edm::Run&,const edm::EventSetup&) ;
      virtual void endRun() ;

      ///Produce the PFRecTrack collection
      virtual void produce(edm::Event&, const edm::EventSetup&);

    
      int FindPfRef(const reco::PFRecTrackCollection & PfRTkColl, 
		    reco::GsfTrack, bool);
      
      bool isFifthStep(reco::PFRecTrackRef pfKfTrack);

      bool applySelection(reco::GsfTrack);
      
      bool resolveGsfTracks(const std::vector<reco::GsfPFRecTrack> &GsfPFVec,
			    unsigned int ngsf,
			    std::vector<unsigned int> &secondaries);

      float selectSecondaries(const reco::GsfPFRecTrack primGsf,
			      const reco::GsfPFRecTrack secGsf); 
      
      // ----------member data ---------------------------
      reco::GsfPFRecTrack pftrack_;
      reco::GsfPFRecTrack secpftrack_;
      edm::ParameterSet conf_;
      edm::InputTag gsfTrackLabel_;
      edm::InputTag pfTrackLabel_;

      ///PFTrackTransformer
      PFTrackTransformer *pfTransformer_;     
      const MultiTrajectoryStateMode *mtsMode_;
      MultiTrajectoryStateTransform  mtsTransform_;

      ///Trajectory of GSfTracks in the event?
      bool trajinev_;
      bool modemomentum_;
      bool applySel_;
      bool applyGsfClean_;
      bool useFifthStep_;
      bool useFifthStepSec_;
      double SCEne_;
      double detaGsfSC_;
      double dphiGsfSC_;

};
#endif
