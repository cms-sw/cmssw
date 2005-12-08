#ifndef ReadRecHit_h
#define ReadRecHit_h

/** \class ReadRecHit
 *
 * ReadRecHit is the EDProducer subclass which finds seeds
 *
 * \author Oliver Gutsche, Fermilab
 *
 * \version   1st Version Aug. 01, 2005  

 *
 ************************************************************/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/EDProduct/interface/EDProduct.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoLocalTracker/ReadRecHit/interface/ReadRecHitAlgorithm.h"

namespace cms
{
  class ReadRecHit : public edm::EDProducer
  {
  public:

    explicit ReadRecHit(const edm::ParameterSet& conf);

    virtual ~ReadRecHit();

    virtual void produce(edm::Event& e, const edm::EventSetup& c);

  private:
    ReadRecHitAlgorithm readRecHitAlgorithm_;
    edm::ParameterSet conf_;

  };
}


#endif
