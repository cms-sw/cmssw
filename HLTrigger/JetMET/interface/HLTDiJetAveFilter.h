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

      double minEtAve_;
      double minEtJet3_;
      double minDphi_;
};

#endif //HLTDiJetAveFilter_h
