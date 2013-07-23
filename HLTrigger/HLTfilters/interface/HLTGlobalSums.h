#ifndef HLTGlobalSums_h
#define HLTGlobalSums_h

/** \class HLTGlobalSums
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing cuts on
 *  global sums such as the scalar sum of Et (a.k.a. H_T), available
 *  in the T=CaloMET or T=MET object.
 *
 *  $Date: 2012/02/24 13:13:46 $
 *  $Revision: 1.8 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include<string>

//
// class declaration
//

template<typename T>
class HLTGlobalSums : public HLTFilter {

   public:

      explicit HLTGlobalSums(const edm::ParameterSet&);
      ~HLTGlobalSums();
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);

   private:
      // configuration
      edm::InputTag inputTag_; // input tag identifying MET product
      int triggerType_;        // triggerType configured
      std::string observable_; // which observable to cut on
      double min_,max_;        // cut: Min<=observable<=Max
      int min_N_;              // how many needed to pass
      int tid_;                // actual triggerType
};

#endif //HLTGlobalSums_h
