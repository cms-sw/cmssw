#ifndef HLTEgammaEtFilterPairs_h
#define HLTEgammaEtFilterPairs_h

/** \class HLTEgammaEtFilterPairs
 *
 *  \author Alessio Ghezzi
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

namespace edm {
  class ConfigurationDescriptions;
}

//
// class decleration
//

class HLTEgammaEtFilterPairs : public HLTFilter {

   public:
      explicit HLTEgammaEtFilterPairs(const edm::ParameterSet&);
      ~HLTEgammaEtFilterPairs();
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

   private:
      edm::InputTag inputTag_; // input tag identifying product contains egammas
      edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> inputToken_;
      double etcutEB1_;           // Et threshold in GeV
      double etcutEB2_;           // Et threshold in GeV
      double etcutEE1_;           // Et threshold in GeV
      double etcutEE2_;           // Et threshold in GeV
      bool   relaxed_;
      edm::InputTag L1IsoCollTag_;
      edm::InputTag L1NonIsoCollTag_;
};

#endif //HLTEgammaEtFilterPairs_h
