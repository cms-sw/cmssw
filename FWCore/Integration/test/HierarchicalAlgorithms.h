#ifndef FWCORE_INTEGRATION_TEST_HIERARCHICAL_ALGIRHTMS_H
#define FWCORE_INTEGRATION_TEST_HIERARCHICAL_ALGIRHTMS_H

/** \class alg_1 and alg_2
 *
 ************************************************************/

#include <vector>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edmtest {

  class alg_2
  {
  public:
    explicit alg_2(const edm::ParameterSet& ps) :
      flavor_( edm::getParameter<std::string>(ps, "flavor")),
      debugLevel_( edm::getUntrackedParameter<int>(ps, "debug", 0) )
    { }

    std::string& flavor() { return flavor_; }

  private:
    std::string flavor_;
    int         debugLevel_;
  };

  class alg_1
  {
  public:
    explicit alg_1(const edm::ParameterSet& ps) : 
      count_( edm::getParameter<int>(ps, "count") ),
      inner_alg_( edm::getParameter<edm::ParameterSet>(ps, "nest_2"))
    { }

  private:
    int   count_;
    alg_2 inner_alg_;
  };

}

#endif
