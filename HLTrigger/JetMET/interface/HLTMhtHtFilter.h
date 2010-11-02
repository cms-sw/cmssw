#ifndef HLTMhtHtFilter_h
#define HLTMhtHtFilter_h

/** \class HLTMhtHtFilter
 *
 *  \author Gheorghe Lungu
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class declaration
//

class HLTMhtHtFilter : public HLTFilter {

   public:
      explicit HLTMhtHtFilter(const edm::ParameterSet&);
      ~HLTMhtHtFilter();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag inputJetTag_; // input tag identifying jets
      bool saveTag_;              // whether to save this tag
      double minMht_;
      double minPtJet_;
      int mode_;
      double etaJet_;
      bool usePt_;
      double minPT12_;
      double minMeff_;
      double minHt_;
};

#endif //HLTMhtHtFilter_h
