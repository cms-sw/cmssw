#ifndef JetProducers_IterativeConeJetProducer_h
#define JetProducers_IterativeConeJetProducer_h

/** \class IterativeConeJetProducer
 *
 * IterativeConeJetProducer is the EDProducer subclass which runs 
 * the CMSIterativeConeAlgorithm jet-finding algorithm.
 *
 ************************************************************/

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoJets/JetAlgorithms/interface/CMSIterativeConeAlgorithm.h"

namespace cms
{
  class IterativeConeJetProducer : public edm::EDProducer
  {
  public:

    // The following is not yet used, but will be the primary
    // constructor when the parameter set system is available.
    //
    explicit IterativeConeJetProducer(const edm::ParameterSet& ps);

    virtual ~IterativeConeJetProducer();

    virtual void produce(edm::Event& e, const edm::EventSetup& c);

  private:
    CMSIterativeConeAlgorithm alg_;
    std::string src_;
    std::string jetType_;
  };
}


#endif
