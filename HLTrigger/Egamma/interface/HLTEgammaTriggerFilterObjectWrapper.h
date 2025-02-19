#ifndef HLTEgammaTriggerFilterObjectWrapper_h
#define HLTEgammaTriggerFilterObjectWrapper_h

/** \class HLTEgammaTriggerFilterObjectWrapper
 *
 *  \author Alessio Ghezzi
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class decleration
//

class HLTEgammaTriggerFilterObjectWrapper : public HLTFilter {

   public:
      explicit HLTEgammaTriggerFilterObjectWrapper(const edm::ParameterSet&);
      ~HLTEgammaTriggerFilterObjectWrapper();
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);

   private:
      edm::InputTag candIsolatedTag_; // input tag identifying product contains egammas
      edm::InputTag candNonIsolatedTag_; // input tag identifying product contains egammas
      bool doIsolated_;
};

#endif //HLTEgammaTriggerFilterObjectWrapper_h
