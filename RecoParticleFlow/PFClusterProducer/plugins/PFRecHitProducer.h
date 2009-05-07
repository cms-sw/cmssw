#ifndef RecoParticleFlow_PFClusterProducer_PFRecHitProducer_h_
#define RecoParticleFlow_PFClusterProducer_PFRecHitProducer_h_

// system include files
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"

/**\class PFRecHitProducer 
\brief Base producer for particle flow rechits (PFRecHit) 

\author Colin Bernet
\date   february 2008
*/



class PFRecHitProducer : public edm::EDProducer {
 public:
  explicit PFRecHitProducer(const edm::ParameterSet&);
  ~PFRecHitProducer();

  
  void produce(edm::Event& iEvent, 
	       const edm::EventSetup& iSetup);



 protected:

  typedef  std::map<unsigned, unsigned >::const_iterator  IDH;

  /// translate the standard rechits ( or calotowers) 
  /// them to PFRecHits, which are stored in the rechits vector
  virtual void createRecHits(std::vector<reco::PFRecHit>& rechits,
			     edm::Event&, const edm::EventSetup&) = 0;  


  // ----------member data ---------------------------
  

  /// verbose ?
  bool   verbose_;

  /// rechits with E < threshold will not give rise to a PFRecHit
  double  thresh_Barrel_;
  double  thresh_Endcap_;
};

#endif
