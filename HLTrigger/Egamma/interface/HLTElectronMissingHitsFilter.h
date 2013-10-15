#ifndef HLTElectronMissingHitsFilter_h
#define HLTElectronMissingHitsFilter_h

/** \class HLTElectronMissingHitsFilter
 *
 *  \author Matteo Sani (UCSD)
 */

namespace edm {
  class ConfigurationDescriptions;
}

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

class HLTElectronMissingHitsFilter : public HLTFilter {
 public:
      explicit HLTElectronMissingHitsFilter(const edm::ParameterSet&);
      ~HLTElectronMissingHitsFilter();
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);
      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:
      edm::InputTag candTag_;            // input tag for the RecoCandidates from the previous filter
      edm::InputTag electronProducer_;   // input tag for the producer of electrons

      int barrelcut_;      // barrel cut
      int endcapcut_;      // endcap cut
      int ncandcut_;       // number of electrons required
};

#endif
