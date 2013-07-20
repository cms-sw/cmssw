#ifndef HLTSinglet_h
#define HLTSinglet_h

/** \class HLTSinglet
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing a basic HLT
 *  trigger for single objects of the same physics type, cutting on
 *  variables relating to their 4-momentum representation
 *
 *  $Date: 2012/02/24 13:13:47 $
 *  $Revision: 1.9 $
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
      ~HLTSinglet();
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);

   private:
      edm::InputTag inputTag_;  // input tag identifying product
      int    triggerType_ ;     // triggerType configured
      double min_E_;            // energy threshold in GeV 
      double min_Pt_;           // pt threshold in GeV 
      double min_Mass_;         // mass threshold in GeV 
      double max_Eta_;          // eta range (symmetric)
      int    min_N_;            // number of objects passing cuts required
      int    tid_;              // actual triggerType
};

#endif // HLTSinglet_h
