//
// HLTFilter selecting events with a minimum number of tri-object candidates passing invariant-mass cuts
//

#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include <string>
#include <vector>

template <typename T1, typename T2, typename T3>
class HLTTripletMass : public HLTFilter {
  typedef edm::Ref<std::vector<T1>> T1Ref;
  typedef edm::Ref<std::vector<T2>> T2Ref;
  typedef edm::Ref<std::vector<T3>> T3Ref;

public:
  explicit HLTTripletMass(const edm::ParameterSet&);
  ~HLTTripletMass() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  bool hltFilter(edm::Event&,
                 const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;
  bool getCollections(edm::Event& iEvent,
                      std::vector<T1Ref>& coll1,
                      std::vector<T2Ref>& coll2,
                      std::vector<T3Ref>& coll3,
                      trigger::TriggerFilterObjectWithRefs& filterproduct) const;

private:
  // configuration
  const std::vector<edm::InputTag> originTag1_;  // input tag identifying originals 1st product
  const std::vector<edm::InputTag> originTag2_;  // input tag identifying originals 2nd product
  const std::vector<edm::InputTag> originTag3_;  // input tag identifying originals 3rd product
  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> inputToken1_;
  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> inputToken2_;
  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> inputToken3_;
  const int triggerType1_;
  const int triggerType2_;
  const int triggerType3_;
  const std::vector<double> min_InvMass_;  // minimum invariant mass of pair
  const std::vector<double> max_InvMass_;  // maximum invariant mass of pair
  const double max_DR_;                    // maximum deltaR between the p4 of 3rd product and p4 of (1+2) product
  const double max_DR2_;                   // maximum deltaR^2 between the p4 of 3rd product and p4 of (1+2) product
  const int min_N_;
  const bool is1and2Same_;
  const bool is2and3Same_;
};

template <typename T1, typename T2, typename T3>
HLTTripletMass<T1, T2, T3>::HLTTripletMass(const edm::ParameterSet& iConfig)
    : HLTFilter(iConfig),
      originTag1_(iConfig.getParameter<std::vector<edm::InputTag>>("originTag1")),
      originTag2_(iConfig.getParameter<std::vector<edm::InputTag>>("originTag2")),
      originTag3_(iConfig.getParameter<std::vector<edm::InputTag>>("originTag3")),
      inputToken1_(consumes(iConfig.getParameter<edm::InputTag>("inputTag1"))),
      inputToken2_(consumes(iConfig.getParameter<edm::InputTag>("inputTag2"))),
      inputToken3_(consumes(iConfig.getParameter<edm::InputTag>("inputTag3"))),
      triggerType1_(iConfig.getParameter<int>("triggerType1")),
      triggerType2_(iConfig.getParameter<int>("triggerType2")),
      triggerType3_(iConfig.getParameter<int>("triggerType3")),
      min_InvMass_(iConfig.getParameter<vector<double>>("MinInvMass")),
      max_InvMass_(iConfig.getParameter<vector<double>>("MaxInvMass")),
      max_DR_(iConfig.getParameter<double>("MaxDR")),
      max_DR2_(max_DR_ * max_DR_),
      min_N_(iConfig.getParameter<int>("MinN")),
      is1and2Same_(iConfig.getParameter<bool>("is1and2Same")),
      is2and3Same_(iConfig.getParameter<bool>("is2and3Same")) {
  if (min_InvMass_.size() != max_InvMass_.size()) {
    throw cms::Exception("Configuration") << "size of \"MinInvMass\" (" << min_InvMass_.size()
                                          << ") and \"MaxInvMass\" (" << max_InvMass_.size() << ") differ";
  }
  if (max_DR_ < 0) {
    throw cms::Exception("Configuration") << "invalid value for parameter \"MaxDR\" (must be >= 0): " << max_DR_;
  }
}

template <typename T1, typename T2, typename T3>
void HLTTripletMass<T1, T2, T3>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<std::vector<edm::InputTag>>("originTag1", {edm::InputTag("hltOriginal1")});
  desc.add<std::vector<edm::InputTag>>("originTag2", {edm::InputTag("hltOriginal2")});
  desc.add<std::vector<edm::InputTag>>("originTag3", {edm::InputTag("hltOriginal3")});
  desc.add<edm::InputTag>("inputTag1", edm::InputTag("hltFiltered1"));
  desc.add<edm::InputTag>("inputTag2", edm::InputTag("hltFiltered2"));
  desc.add<edm::InputTag>("inputTag3", edm::InputTag("hltFiltered3"));
  desc.add<int>("triggerType1", 0);
  desc.add<int>("triggerType2", 0);
  desc.add<int>("triggerType3", 0);

  desc.add<vector<double>>("MinInvMass", {0});
  desc.add<vector<double>>("MaxInvMass", {1e12});

  desc.add<double>("MaxDR", 1e4);
  desc.add<int>("MinN", 0);

  desc.add<bool>("is1and2Same", false);
  desc.add<bool>("is2and3Same", false);
  descriptions.addWithDefaultLabel(desc);
}

template <typename T1, typename T2, typename T3>
bool HLTTripletMass<T1, T2, T3>::getCollections(edm::Event& iEvent,
                                                std::vector<T1Ref>& coll1,
                                                std::vector<T2Ref>& coll2,
                                                std::vector<T3Ref>& coll3,
                                                trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  edm::Handle<trigger::TriggerFilterObjectWithRefs> handle1, handle2, handle3;
  if (iEvent.getByToken(inputToken1_, handle1) and iEvent.getByToken(inputToken2_, handle2) and
      iEvent.getByToken(inputToken3_, handle3)) {
    // get hold of pre-filtered object collections
    handle1->getObjects(triggerType1_, coll1);
    handle2->getObjects(triggerType2_, coll2);
    handle3->getObjects(triggerType3_, coll3);
    const trigger::size_type n1(coll1.size());
    const trigger::size_type n2(coll2.size());
    const trigger::size_type n3(coll3.size());

    if (saveTags()) {
      edm::InputTag tagOld;
      for (unsigned int i = 0; i < originTag1_.size(); ++i) {
        filterproduct.addCollectionTag(originTag1_[i]);
      }
      tagOld = edm::InputTag();
      for (trigger::size_type i1 = 0; i1 != n1; ++i1) {
        const edm::ProductID pid(coll1[i1].id());
        const auto& prov = iEvent.getStableProvenance(pid);
        const std::string& label(prov.moduleLabel());
        const std::string& instance(prov.productInstanceName());
        const std::string& process(prov.processName());
        edm::InputTag tagNew(edm::InputTag(label, instance, process));
        if (tagOld.encode() != tagNew.encode()) {
          filterproduct.addCollectionTag(tagNew);
          tagOld = tagNew;
        }
      }
      for (unsigned int i = 0; i < originTag2_.size(); ++i) {
        filterproduct.addCollectionTag(originTag2_[i]);
      }
      tagOld = edm::InputTag();
      for (trigger::size_type i2 = 0; i2 != n2; ++i2) {
        const edm::ProductID pid(coll2[i2].id());
        const auto& prov = iEvent.getStableProvenance(pid);
        const std::string& label(prov.moduleLabel());
        const std::string& instance(prov.productInstanceName());
        const std::string& process(prov.processName());
        edm::InputTag tagNew(edm::InputTag(label, instance, process));
        if (tagOld.encode() != tagNew.encode()) {
          filterproduct.addCollectionTag(tagNew);
          tagOld = tagNew;
        }
      }
      for (unsigned int i = 0; i < originTag3_.size(); ++i) {
        filterproduct.addCollectionTag(originTag3_[i]);
      }
      tagOld = edm::InputTag();
      for (trigger::size_type i3 = 0; i3 != n3; ++i3) {
        const edm::ProductID pid(coll3[i3].id());
        const auto& prov = iEvent.getStableProvenance(pid);
        const std::string& label(prov.moduleLabel());
        const std::string& instance(prov.productInstanceName());
        const std::string& process(prov.processName());
        edm::InputTag tagNew(edm::InputTag(label, instance, process));
        if (tagOld.encode() != tagNew.encode()) {
          filterproduct.addCollectionTag(tagNew);
          tagOld = tagNew;
        }
      }
    }

    return true;
  } else
    return false;
}

// ------------ method called to produce the data  ------------
template <typename T1, typename T2, typename T3>
bool HLTTripletMass<T1, T2, T3>::hltFilter(edm::Event& iEvent,
                                           const edm::EventSetup& iSetup,
                                           trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  // All HLT filters must create and fill an HLT filter object,
  // recording any reconstructed physics objects satisfying (or not)
  // this HLT filter, and place it in the Event.

  std::vector<T1Ref> coll1;
  std::vector<T2Ref> coll2;
  std::vector<T3Ref> coll3;

  int n(0);
  if (getCollections(iEvent, coll1, coll2, coll3, filterproduct)) {
    T1Ref r1;
    T2Ref r2;
    T3Ref r3;

    reco::Particle::LorentzVector dauA_p4, dauB_p4, dauAB_p4, dauC_p4;
    for (unsigned int i1 = 0; i1 != coll1.size(); i1++) {
      r1 = coll1[i1];
      dauA_p4 = reco::Particle::LorentzVector(r1->px(), r1->py(), r1->pz(), r1->energy());
      unsigned int i2 = is1and2Same_ ? i1 + 1 : 0;
      for (; i2 != coll2.size(); i2++) {
        r2 = coll2[i2];
        dauB_p4 = reco::Particle::LorentzVector(r2->px(), r2->py(), r2->pz(), r2->energy());
        dauAB_p4 = dauA_p4 + dauB_p4;

        unsigned int i3 = is2and3Same_ ? i2 + 1 : 0;
        for (; i3 != coll3.size(); i3++) {
          r3 = coll3[i3];
          dauC_p4 = reco::Particle::LorentzVector(r3->px(), r3->py(), r3->pz(), r3->energy());
          if (reco::deltaR2(dauAB_p4, dauC_p4) > max_DR2_) {
            continue;
          }
          bool passesMassCut = false;
          auto const mass_ABC = (dauC_p4 + dauAB_p4).mass();
          for (unsigned int j = 0; j < max_InvMass_.size(); j++) {
            if ((mass_ABC >= min_InvMass_[j]) and (mass_ABC < max_InvMass_[j])) {
              passesMassCut = true;
              break;
            }
          }
          if (passesMassCut) {
            n++;
            filterproduct.addObject(triggerType1_, r1);
            filterproduct.addObject(triggerType2_, r2);
            filterproduct.addObject(triggerType3_, r3);
          }
        }
      }
    }
  }

  return (n >= min_N_);
}

typedef HLTTripletMass<reco::RecoChargedCandidate, reco::RecoChargedCandidate, reco::RecoEcalCandidate>
    HLT3MuonMuonPhotonMass;

DEFINE_FWK_MODULE(HLT3MuonMuonPhotonMass);
