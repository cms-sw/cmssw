#ifndef JetProducers_MidpointJetProducer_h
#define JetProducers_MidpointJetProducer_h

/** \class MidpointJetProducer
 *
 * MidpointJetProducer is the EDProducer subclass which runs 
 * the CMSmidpointAlgorithm jet-finding algorithm.
 *
 * \author Marc Paterno, Fermilab
 *
 * \version   1st Version Apr. 6, 2005  

 *
 ************************************************************/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/EDProduct/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/JetExamples/interface/CMSmidpointAlgorithm.h"

namespace cms
{
  class MidpointJetProducer : public edm::EDProducer {
  public:
    explicit MidpointJetProducer(const edm::ParameterSet& ps);
    virtual ~MidpointJetProducer();
    virtual void produce(edm::Event& e, const edm::EventSetup& c);

  private:
    CMSmidpointAlgorithm alg_;
    std::string src_;
  };
}


#endif
