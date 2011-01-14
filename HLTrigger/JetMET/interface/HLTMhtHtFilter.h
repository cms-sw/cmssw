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
      std::vector<double> minPtJet_;
      int minNJet_;
      int mode_;
      std::vector<double> etaJet_;
      bool usePt_;
      double minPT12_;
      double minMeff_;
      double minHt_;
      double minMht2Ht_;
};

#endif //HLTMhtHtFilter_h
