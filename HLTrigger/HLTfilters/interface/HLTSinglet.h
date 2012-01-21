#ifndef HLTSinglet_h
#define HLTSinglet_h

/** \class HLTSinglet
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing a basic HLT
 *  trigger for single objects of the same physics type, cutting on
 *  variables relating to their 4-momentum representation
 *
 *  $Date: 2011/07/12 17:35:52 $
 *  $Revision: 1.5 $
 *
 *  \author Martin Grunewald
 *
 */

#include<vector>
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class declaration
//

template<typename T, int Tid>
class HLTSinglet : public HLTFilter {

   public:
      explicit HLTSinglet(const edm::ParameterSet&);
      ~HLTSinglet();
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);

   private:
      edm::InputTag inputTag_;  // input tag identifying product
      double min_Pt_;           // pt threshold in GeV 
      double max_Eta_;          // eta range (symmetric)
      int    min_N_;            // number of objects passing cuts required
};

#endif // HLTSinglet_h
