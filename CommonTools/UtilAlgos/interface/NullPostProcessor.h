#ifndef UtilAlgos_NullPostProcessor_h
#define UtilAlgos_NullPostProcessor_h
/* \class helper::NullPostProcessor<OutputCollection>
 *
 * \author Luca Lista, INFN
 */
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

namespace edm {
  class EDFilter;
  class Event;
  class ParameterSet;
}  // namespace edm

namespace helper {

  template <typename OutputCollection, typename EdmFilter = edm::EDFilter>
  struct NullPostProcessor {
    NullPostProcessor(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC) : NullPostProcessor(iConfig) {}
    NullPostProcessor(const edm::ParameterSet& iConfig) {}
    void init(EdmFilter&) {}
    void process(edm::OrphanHandle<OutputCollection>, edm::Event&) {}
  };

}  // namespace helper

#endif
