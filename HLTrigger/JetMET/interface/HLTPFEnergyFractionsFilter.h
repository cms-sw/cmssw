#ifndef HLTPFEnergyFractionsFilter_h
#define HLTPFEnergyFractionsFilter_h

/** \class HLTPFEnergyFractionsFilter
 *
 *  \author Srimanobhas Phat
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

namespace edm {
   class ConfigurationDescriptions;
}


//
// class declaration
//

class HLTPFEnergyFractionsFilter : public HLTFilter {

   public:
      explicit HLTPFEnergyFractionsFilter(const edm::ParameterSet&);
      ~HLTPFEnergyFractionsFilter();
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);

   private:
      edm::InputTag inputPFJetTag_;   // input tag identifying pfjets
      unsigned int nJet_;           // No. of jet to check with this filter 
      double min_CEEF_;
      double max_CEEF_;
      double min_NEEF_;
      double max_NEEF_;
      double min_CHEF_;
      double max_CHEF_;
      double min_NHEF_;
      double max_NHEF_;
};

#endif //HLTPFEnergyFractionsFilter_h
