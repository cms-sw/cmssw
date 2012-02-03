#ifndef HLTAlphaTFilter_h
#define HLTAlphaTFilter_h

/** \class HLTAlphaTFilter
 *
 *  \author Bryn Mathias
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

namespace edm {
   class ConfigurationDescriptions;
}


//
// class declaration
//

class HLTAlphaTFilter : public edm::EDFilter {

   public:
      explicit HLTAlphaTFilter(const edm::ParameterSet&);
      ~HLTAlphaTFilter();
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      
   private:
      edm::InputTag inputJetTag_; // input tag identifying jets
      edm::InputTag inputJetTagFastJet_; // input tag identifying a second collection of jets
      std::vector<double> minPtJet_;
      std::vector<double> etaJet_;
      double minHt_;
      double minAlphaT_;
};

#endif //HLTAlphaTFilter_h
