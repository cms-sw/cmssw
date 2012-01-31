#ifndef HLTFatJetMassFilter_h
#define HLTFatJetMassFilter_h

/** \class HLTFatJetMassFilter
 *
 *  \author Maurizio Pierini
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

namespace edm {
   class ConfigurationDescriptions;
}


//
// class declaration
//

class HLTFatJetMassFilter : public HLTFilter {

   public:
      explicit HLTFatJetMassFilter(const edm::ParameterSet&);
      ~HLTFatJetMassFilter();
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);

   private:
      edm::InputTag inputJetTag_; // input tag identifying jets
      double minMass_;
      double fatJetDeltaR_;
      double maxDeltaEta_;
      double maxJetEta_;
      double minJetPt_;
};

#endif //HLTFatJetMassFilter_h
