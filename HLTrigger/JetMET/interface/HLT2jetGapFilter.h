#ifndef HLT2jetGapFilter_h
#define HLT2jetGapFilter_h

/** \class HLT2jetGapFilter
 *
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class declaration
//

class HLT2jetGapFilter : public HLTFilter {

   public:
      explicit HLT2jetGapFilter(const edm::ParameterSet&);
      ~HLT2jetGapFilter();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag inputTag_; // input tag identifying jets
      bool saveTags_;           // whether to save this tag
      double minEt_;
      double minEta_;
};

#endif //HLT2jetGapFilter_h
