#ifndef HLTMonoJetFilter_h
#define HLTMonoJetFilter_h

/** \class HLTMonoJetFilter
 *
 *  \author Srimanobhas Phat
 *
 */

#include<vector>
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"


//
// class declaration
//

template<typename T, int Tid>
class HLTMonoJetFilter : public HLTFilter {

   public:
      explicit HLTMonoJetFilter(const edm::ParameterSet&);
      ~HLTMonoJetFilter();
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);

   private:
      edm::InputTag inputJetTag_;   // input tag identifying jets
      double max_PtSecondJet_;
      double max_DeltaPhi_;
};

#endif //HLTMonoJetFilter_h
