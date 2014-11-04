#ifndef HLTMuonGenericFilter_h
#define HLTMuonGenericFilter_h

/** \class HLTMuonGenericFilter
 *
 *  \author Roberto Covarelli (CERN)
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateIsolation.h"

namespace edm {
  class ConfigurationDescriptions;
}

//
// class declaration
//

class HLTMuonGenericFilter : public HLTFilter {

   public:
      explicit HLTMuonGenericFilter(const edm::ParameterSet&);
      ~HLTMuonGenericFilter();
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

   private:
      edm::InputTag candTag_; // input tag identifying product that contains filtered photons
      edm::InputTag isoTag_; // input tag identifying product that contains isolated map
      edm::InputTag nonIsoTag_; // input tag identifying product that contains non-isolated map
      edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> candToken_;
      edm::EDGetTokenT<reco::RecoChargedCandidateIsolationMap> isoToken_;
      edm::EDGetTokenT<reco::RecoChargedCandidateIsolationMap> nonIsoToken_;
      bool lessThan_;           // the cut is "<" or ">" ?
      bool useEt_;              // use E or Et in relative isolation cuts
      double thrRegularEB_;     // threshold for regular cut (x < thr) - ECAL barrel
      double thrRegularEE_;     // threshold for regular cut (x < thr) - ECAL endcap
      double thrOverEEB_;       // threshold for x/E < thr cut (isolations) - ECAL barrel
      double thrOverEEE_;       // threshold for x/E < thr cut (isolations) - ECAL endcap
      double thrOverE2EB_;      // threshold for x/E^2 < thr cut (isolations) - ECAL barrel
      double thrOverE2EE_;      // threshold for x/E^2 < thr cut (isolations) - ECAL endcap
      int    ncandcut_;        // number of photons required
      bool doIsolated_;

      edm::InputTag L1IsoCollTag_;
      edm::InputTag L1NonIsoCollTag_;
};

#endif //HLTMuonGenericFilter_h


