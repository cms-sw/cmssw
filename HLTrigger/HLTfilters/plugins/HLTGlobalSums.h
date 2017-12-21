#ifndef HLTGlobalSums_h
#define HLTGlobalSums_h

/** \class HLTGlobalSums
 *
 *
 *  This class is an HLTFilter (-> EDFilter) implementing cuts on
 *  global sums such as the scalar sum of Et (a.k.a. H_T), available
 *  in the T=CaloMET or T=MET object.
 *
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include<string>
#include<vector>

//
// class declaration
//

template<typename T>
class HLTGlobalSums : public HLTFilter {

   public:

      explicit HLTGlobalSums(const edm::ParameterSet&);
      ~HLTGlobalSums() override;
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;

   private:
      // configuration
      edm::InputTag                     inputTag_;   // input tag identifying MET product
      edm::EDGetTokenT<std::vector<T> > inputToken_; // token identifying MET product
      int triggerType_;        // triggerType configured
      std::string observable_; // which observable to cut on
      double min_,max_;        // cut: Min<=observable<=Max
      int min_N_;              // how many needed to pass
      int tid_;                // actual triggerType
};

#endif //HLTGlobalSums_h
