#ifndef HLTRapGapFilter_h
#define HLTRapGapFilter_h

/** \class HLTRapGapFilter
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class decleration
//

class HLTRapGapFilter : public HLTFilter {

   public:
      explicit HLTRapGapFilter(const edm::ParameterSet&);
      ~HLTRapGapFilter();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag inputTag_; // input tag identifying jets
      bool saveTags_;           // whether to save this tag
      double absEtaMin_;
      double absEtaMax_;
      double caloThresh_;
};

#endif //HLTRapGapFilter_h
