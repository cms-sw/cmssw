#ifndef HLTSummaryFilter_h
#define HLTSummaryFilter_h

/** \class HLTSummaryFilter
 *
 *
 *  This class is an HLTFilter (-> EDFilter) implementing a smart HLT
 *  trigger cut, specified as a string such as "pt>15 && -3<eta<3",
 *  for objects in the TriggerSummaryAOD product, allowing to cut on
 *  variables relating to their 4-momentum representation
 *
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include<string>

namespace edm {
  class ConfigurationDescriptions;
}

//
// class declaration
//

class HLTSummaryFilter : public HLTFilter {

   public:

      explicit HLTSummaryFilter(const edm::ParameterSet&);
      ~HLTSummaryFilter();
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;

   private:
      edm::InputTag                           summaryTag_;   // input tag identifying TriggerSummaryAOD
      edm::EDGetTokenT<trigger::TriggerEvent> summaryToken_; // token identifying TriggerSummaryAOD
      edm::InputTag memberTag_;  // which packed-up collection or filter
      std::string   cut_;        // smart cut
      int           min_N_;      // number of objects passing cuts required

      StringCutObjectSelector<trigger::TriggerObject> select_; // smart selector
};

#endif //HLTSummaryFilter_h
