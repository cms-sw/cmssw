#ifndef HLTExclDiJetFilter_h
#define HLTExclDiJetFilter_h

/** \class HLTExclDiJetFilter
 *
 *  \author Dominique J. Mangeol
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

namespace edm {
   class ConfigurationDescriptions;
}

//
// class declaration
//

class HLTExclDiJetFilter : public HLTFilter {

   public:
      explicit HLTExclDiJetFilter(const edm::ParameterSet&);
      ~HLTExclDiJetFilter();
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag inputJetTag_; // input tag identifying jets
      bool saveTag_;              // whether to save this tag
      double minPtJet_;
      double minHFe_;
      bool HF_OR_;
};

#endif //HLTExclDiJetFilter_h
