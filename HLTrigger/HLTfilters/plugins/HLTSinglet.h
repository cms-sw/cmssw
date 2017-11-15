#ifndef HLTSinglet_h
#define HLTSinglet_h

/** \class HLTSinglet
 *
 *
 *  This class is an HLTFilter (-> EDFilter) implementing a basic HLT
 *  trigger for single objects of the same physics type, cutting on
 *  variables relating to their 4-momentum representation
 *
 *
 *  \author Martin Grunewald
 *
 */

#include<vector>
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

//
// class declaration
//

template<typename T>
class HLTSinglet : public HLTFilter {

   public:
      explicit HLTSinglet(const edm::ParameterSet&);
      ~HLTSinglet() override;
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;

   private:
      const edm::InputTag                    inputTag_;     // input tag identifying product
      const edm::EDGetTokenT<std::vector<T>> inputToken_;   // token identifying product
      const int    triggerType_ ;                           // triggerType configured
      const int    min_N_;                                  // number of objects passing cuts required
      const double min_E_;                                  // energy threshold in GeV
      const double min_Pt_;                                 // pt threshold in GeV
      const double min_Mass_;                               // min mass threshold in GeV
      const double max_Mass_;                               // max mass threshold in GeV
      const double min_Eta_;                                // lower eta cut to define eta-range (symmetric)
      const double max_Eta_;                                // upper eta cut to define eta-range (symmetric)
};

#endif // HLTSinglet_h
