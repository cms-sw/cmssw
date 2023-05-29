#ifndef Integration_HierarchicalEDProducer_h
#define Integration_HierarchicalEDProducer_h

/** \class HierarchicalEDProducer
 *
 * \version   1st Version Apr. 6, 2005  

 *
 ************************************************************/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "HierarchicalAlgorithms.h"

namespace edmtest {
  class HierarchicalEDProducer : public edm::global::EDProducer<> {
  public:
    explicit HierarchicalEDProducer(edm::ParameterSet const& ps);

    ~HierarchicalEDProducer() override;

    void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const override;

  private:
    const double radius_;
    const alg_1 outer_alg_;
  };
}  // namespace edmtest
#endif
