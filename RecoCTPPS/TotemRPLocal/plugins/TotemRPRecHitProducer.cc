/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
* 	Hubert Niewiadomski
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/CTPPSReco/interface/TotemRPCluster.h"
#include "DataFormats/CTPPSReco/interface/TotemRPRecHit.h"

#include "RecoCTPPS/TotemRPLocal/interface/TotemRPRecHitProducerAlgorithm.h"
 
//----------------------------------------------------------------------------------------------------

class TotemRPRecHitProducer : public edm::stream::EDProducer<>
{
  public:
  
    explicit TotemRPRecHitProducer(const edm::ParameterSet& conf);
  
    virtual ~TotemRPRecHitProducer() {}
  
    virtual void produce(edm::Event& e, const edm::EventSetup& c) override;
  
  private:
    const edm::ParameterSet conf_;
    int verbosity_;

    TotemRPRecHitProducerAlgorithm algorithm_;

    edm::InputTag tagCluster_;
    edm::EDGetTokenT<edm::DetSetVector<TotemRPCluster>> tokenCluster_;
};

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

TotemRPRecHitProducer::TotemRPRecHitProducer(const edm::ParameterSet& conf) :
  conf_(conf), algorithm_(conf)
{
  verbosity_ = conf.getParameter<int>("verbosity");

  tagCluster_ = conf.getParameter<edm::InputTag>("tagCluster");
  tokenCluster_ = consumes<edm::DetSetVector<TotemRPCluster> >(tagCluster_);

  produces<edm::DetSetVector<TotemRPRecHit>>();
}

//----------------------------------------------------------------------------------------------------
 
void TotemRPRecHitProducer::produce(edm::Event& e, const edm::EventSetup& es)
{
  // get input
  edm::Handle< edm::DetSetVector<TotemRPCluster> > input;
  e.getByToken(tokenCluster_, input);
 
  // prepare output
  DetSetVector<TotemRPRecHit> output;

  // build reco hits
  for (auto &ids : *input)
  {
    DetSet<TotemRPRecHit> &ods = output.find_or_insert(ids.detId());
    algorithm_.buildRecoHits(ids, ods);
  }
   
  // save output
  e.put(make_unique<DetSetVector<TotemRPRecHit>>(output));
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(TotemRPRecHitProducer);
