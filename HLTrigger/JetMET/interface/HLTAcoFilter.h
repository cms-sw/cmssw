#ifndef HLTAcoFilter_h
#define HLTAcoFilter_h

/** \class HLTAcoFilter
 *
 *  \author Dominique J. Mangeol
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include <string>
#include <string.h>
#include <cmath>
//
// class declaration
//

class HLTAcoFilter : public HLTFilter {

   public:
      explicit HLTAcoFilter(const edm::ParameterSet&);
      ~HLTAcoFilter();
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);

   private:
      edm::InputTag inputJetTag_; // input tag identifying jets
      edm::InputTag inputMETTag_; // input tag identifying for MET
      double minEtjet1_;
      double minEtjet2_;
      double minDPhi_;
      double maxDPhi_;
      std::string AcoString_;
};

#endif //HLTAcoFilter_h
