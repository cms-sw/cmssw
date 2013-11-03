#ifndef HLTEgammaCaloIsolFilterPairs_h
#define HLTEgammaCaloIsolFilterPairs_h

/** \class HLTEgammaCaloIsolFilterPairs
 *
 *  \author Alessio Ghezzi
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"



//
// class decleration
//

namespace edm {
  class ConfigurationDescriptions;
}

class HLTEgammaCaloIsolFilterPairs : public HLTFilter {

  public:
    explicit HLTEgammaCaloIsolFilterPairs(const edm::ParameterSet&);
    ~HLTEgammaCaloIsolFilterPairs();
    virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;
    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

  private:
    edm::InputTag candTag_; // input tag identifying product contains filtered egammas
    edm::InputTag isoTag_; // input tag identifying product contains ecal isolation map
    edm::InputTag nonIsoTag_; // input tag identifying product contains ecal isolation map
    edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> candToken_;
    edm::EDGetTokenT<reco::RecoEcalCandidateIsolationMap> isoToken_;
    edm::EDGetTokenT<reco::RecoEcalCandidateIsolationMap> nonIsoToken_;

    double isolcut_EB1;
    double FracCut_EB1;
    double IsoloEt2_EB1;
    double isolcut_EE1;
    double FracCut_EE1;
    double IsoloEt2_EE1;

    double isolcut_EB2;
    double FracCut_EB2;
    double IsoloEt2_EB2;
    double isolcut_EE2;
    double FracCut_EE2;
    double IsoloEt2_EE2;

    bool AlsoNonIso_1, AlsoNonIso_2;

    bool PassCaloIsolation(edm::Ref<reco::RecoEcalCandidateCollection> ref,const reco::RecoEcalCandidateIsolationMap& IsoMap,const reco::RecoEcalCandidateIsolationMap& NonIsoMap, int which, bool ChekAlsoNonIso) const;
};

#endif //HLTEgammaCaloIsolFilterPairs_h


