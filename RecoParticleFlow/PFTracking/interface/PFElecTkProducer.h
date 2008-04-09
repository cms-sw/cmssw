#ifndef PFElecTkProducer_H
#define PFElecTkProducer_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

class PFTrackTransformer;
class GsfTrack;



/// \brief Abstract
/*!
\author Michele Pioppi
\date January 2007

 PFElecTkProducer reads GsfTracks collection
 built with the seeds saved by module GoodSeedProducer
 and transform them in PFRecTracks.
*/

class PFElecTkProducer : public edm::EDProducer {
 public:
  
     ///Constructor
     explicit PFElecTkProducer(const edm::ParameterSet&);

     ///Destructor
     ~PFElecTkProducer();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void endJob() ;

      ///Produce the PFRecTrack collection
      virtual void produce(edm::Event&, const edm::EventSetup&);

    
      int FindPfRef(const reco::PFRecTrackCollection & PfRTkColl, 
		    reco::GsfTrack, bool);
 
      bool otherElId(const reco::GsfTrackCollection  & GsfColl, 
		     reco::GsfTrack GsfTk);
	
      // ----------member data ---------------------------
      reco::GsfPFRecTrack pftrack_;
      edm::ParameterSet conf_;
      edm::InputTag gsfTrackLabel_;
      edm::InputTag pfTrackLabel_;

      ///PFTrackTransformer
      PFTrackTransformer *pfTransformer_; 

      ///Trajectory of GSfTracks in the event?
      bool trajinev_;
      bool modemomentum_;
};
#endif
