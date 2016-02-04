#ifndef FWLite_VectorIntProducer_h
#define FWLite_VectorIntProducer_h

/** \class VectorIntProducer
 *
 * \version
 *
 ************************************************************/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

namespace edmtest {
  class VectorIntProducer : public edm::EDProducer {

  public:
    explicit VectorIntProducer(edm::ParameterSet const& ps);
    virtual ~VectorIntProducer();
    virtual void produce(edm::Event& e, edm::EventSetup const& c);

  private:
  };
}
#endif
