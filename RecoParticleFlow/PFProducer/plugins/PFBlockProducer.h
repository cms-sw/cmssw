#ifndef RecoParticleFlow_PFProducer_PFBlockProducer_h_
#define RecoParticleFlow_PFProducer_PFBlockProducer_h_

#include "RecoParticleFlow/PFProducer/interface/PFBlockAlgo.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/Exception.h"

/**\class PFBlockProducer 
\brief Producer for particle flow blocks

This producer makes use of PFBlockAlgo, the particle flow block algorithm.
Particle flow itself consists in reconstructing particles from the particle 
flow blocks This is done at a later stage, see PFProducer and PFAlgo.

\author Colin Bernet
\date   April 2007
*/

class FSimEvent;



class PFBlockProducer : public edm::stream::EDProducer<> {
 public:

  explicit PFBlockProducer(const edm::ParameterSet&);

  ~PFBlockProducer() override;
  
  void beginLuminosityBlock(edm::LuminosityBlock const&, 
				    edm::EventSetup const&) override;

  void produce(edm::Event&, const edm::EventSetup&) override;

 private:
  /// verbose ?
  const bool   verbose_;
  const edm::EDPutTokenT<reco::PFBlockCollection> putToken_;
  
  /// Particle flow block algorithm 
  PFBlockAlgo            pfBlockAlgo_;

};

DEFINE_FWK_MODULE(PFBlockProducer);

#endif
