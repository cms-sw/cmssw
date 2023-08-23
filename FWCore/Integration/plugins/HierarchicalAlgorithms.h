#ifndef Integration_HierarchicalAlgorithms_h
#define Integration_HierarchicalAlgorithms_h

/** \class alg_1 and alg_2
 *
 ************************************************************/

#include <vector>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edmtest {

  class alg_2 {
  public:
    explicit alg_2(const edm::ParameterSet& ps)
        : flavor_(ps.getParameter<std::string>("flavor")), debugLevel_(ps.getUntrackedParameter<int>("debug", 0)) {}

    std::string& flavor() { return flavor_; }

  private:
    std::string flavor_;
    int debugLevel_;
  };

  class alg_1 {
  public:
    explicit alg_1(const edm::ParameterSet& ps)
        : count_(ps.getParameter<int>("count")), inner_alg_(ps.getParameterSet("nest_2")) {}

  private:
    int count_;
    alg_2 inner_alg_;
  };

}  // namespace edmtest

#endif
