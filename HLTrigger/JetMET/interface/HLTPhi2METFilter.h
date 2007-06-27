#ifndef HLTPhi2METFilter_h
#define HLTPhi2METFilter_h

/** \class HLTPhi2METFilter
 *
 *  \author Dominique J. Mangeol
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class declaration
//

class HLTPhi2METFilter : public HLTFilter {

   public:
      explicit HLTPhi2METFilter(const edm::ParameterSet&);
      ~HLTPhi2METFilter();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag inputJetTag_; // input tag identifying jets
      edm::InputTag inputMETTag_; // input tag identifying for MET

      double minEtjet1_;
      double minEtjet2_;
      double minDPhi_;
      double maxDPhi_;
};

#endif //HLTPhi2METFilter_h
