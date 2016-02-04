#ifndef HLTDiJetAveFilter_h
#define HLTDiJetAveFilter_h

/** \class HLTDiJetAveFilter
 *
 *  \author Dominique J. Mangeol
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class declaration
//

class HLTDiJetAveFilter : public HLTFilter {

   public:
      explicit HLTDiJetAveFilter(const edm::ParameterSet&);
      ~HLTDiJetAveFilter();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag inputJetTag_; // input tag identifying jets
      bool saveTag_;              // whether to save this tag
      double minPtAve_;
      double minPtJet3_;
      double minDphi_;
};

#endif //HLTDiJetAveFilter_h
