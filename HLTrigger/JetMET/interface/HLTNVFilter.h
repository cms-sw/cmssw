#ifndef HLTNVFilter_h
#define HLTNVFilter_h

/** \class HLTNVFilter
 *
 *  \author Dominique J. Mangeol
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class declaration
//

class HLTNVFilter : public HLTFilter {

   public:
      explicit HLTNVFilter(const edm::ParameterSet&);
      ~HLTNVFilter();
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);

   private:
      edm::InputTag inputJetTag_; // input tag identifying jets
      edm::InputTag inputMETTag_; // input tag identifying for MET
      double minEtjet1_;
      double minEtjet2_;
      double minNV_;
};

#endif //HLTNVFilter_h
