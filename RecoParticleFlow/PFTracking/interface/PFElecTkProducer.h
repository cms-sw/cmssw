#ifndef PFElecTkProducer_H
#define PFElecTkProducer_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"


class PFTrackTransformer;
class GsfTrack;
class PFRecTrack;

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

 
      // ----------member data ---------------------------
  
      edm::ParameterSet conf_;
      std::string gsfTrackModule_;

      ///PFTrackTransformer
      PFTrackTransformer *pfTransformer_; 

      ///Trajectory of GSfTracks in the event?
      bool trajinev_;
};
#endif
