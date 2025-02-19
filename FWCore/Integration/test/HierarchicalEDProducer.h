#ifndef Integration_HierarchicalEDProducer_h
#define Integration_HierarchicalEDProducer_h

/** \class HierarchicalEDProducer
 *
 * \version   1st Version Apr. 6, 2005  

 *
 ************************************************************/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Integration/test/HierarchicalAlgorithms.h"


namespace edmtest {
  class HierarchicalEDProducer : public edm::EDProducer {
  public:

    explicit HierarchicalEDProducer(edm::ParameterSet const& ps);

    virtual ~HierarchicalEDProducer();

    virtual void produce(edm::Event& e, edm::EventSetup const& c);

  private:
    double       radius_;
    alg_1        outer_alg_;
  };
}
#endif
