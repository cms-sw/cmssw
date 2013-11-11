#ifndef HLTEgammaDoubleEtDeltaPhiFilter_h
#define HLTEgammaDoubleEtDeltaPhiFilter_h

/** \class HLTEgammaDoubleEtDeltaPhiFilter
 *
 *  \author Li Wenbo (PKU)
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

class HLTEgammaDoubleEtDeltaPhiFilter : public HLTFilter {

   public:
      explicit HLTEgammaDoubleEtDeltaPhiFilter(const edm::ParameterSet&);
      ~HLTEgammaDoubleEtDeltaPhiFilter();
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

   private:
      edm::InputTag inputTag_; // input tag identifying product contains filtered candidates
      edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs>  inputToken_;
      double etcut_;           // Et threshold in GeV
      double minDeltaPhi_;    // minimum deltaPhi
 //   int    ncandcut_;        // number of egammas required
      bool   relaxed_;
      edm::InputTag L1IsoCollTag_;
      edm::InputTag L1NonIsoCollTag_;
};

#endif //HLTEgammaDoubleEtDeltaPhiFilter_h
