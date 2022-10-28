#ifndef HLTrigger_HLTfilters_plugins_HLTDoubletSinglet_h
#define HLTrigger_HLTfilters_plugins_HLTDoubletSinglet_h

/** \class HLTDoubletSinglet
 *
 *
 *  This class is an HLTFilter (-> EDFilter) implementing a basic HLT
 *  trigger for triplets of objects, evaluating all triplets with the first
 *  object from collection 1, the second object from collection 2,
 *  and the third object from collection 3, 
 *  cutting on variables relating to their 4-momentum representations.
 *  The filter itself compares only objects from collection 3
 *  with objects from collections 1 and 2.
 *  The object collections are assumed to be outputs of HLTSinglet
 *  single-object-type filters so that the access is thorugh
 *  RefToBases and polymorphic.
 *
 *
 *  \author Jaime Leon Holgado 
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
#include <vector>
#include <cmath>

//
// class declaration
//

template <typename T1, typename T2, typename T3>
class HLTDoubletSinglet : public HLTFilter {
public:
  explicit HLTDoubletSinglet(const edm::ParameterSet&);
  ~HLTDoubletSinglet() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  bool hltFilter(edm::Event&,
                 const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;

private:
  // configuration
  const std::vector<edm::InputTag> originTag1_;  // input tag identifying originals 1st product
  const std::vector<edm::InputTag> originTag2_;  // input tag identifying originals 2nd product
  const std::vector<edm::InputTag> originTag3_;  // input tag identifying originals 3rd product
  const edm::InputTag inputTag1_;                // input tag identifying filtered 1st product
  const edm::InputTag inputTag2_;                // input tag identifying filtered 2nd product
  const edm::InputTag inputTag3_;                // input tag identifying filtered 3rd product
  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> inputToken1_;
  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> inputToken2_;
  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> inputToken3_;
  const int triggerType1_;
  const int triggerType2_;
  const int triggerType3_;
  const double min_Dphi_, max_Dphi_;  // DeltaPhi (1,3) and (2,3) window
  const double min_Deta_, max_Deta_;  // DeltaEta (1,3) and (2,3) window
  const double min_Minv_, max_Minv_;  // Minv(1,2) and Minv(2,3) window
  const double min_DelR_, max_DelR_;  // DeltaR (1,3) and (2,3) window
  const double min_Pt_, max_Pt_;      // Pt(1,3) and (2,3) window
  const int min_N_;                   // number of triplets passing cuts required

  // calculated from configuration in c'tor
  const bool same12_, same13_, same23_;                       // 1st and 2nd product are one and the same
  const double min_DelR2_, max_DelR2_;                        // DeltaR (1,3) and (2,3) window
  const bool cutdphi_, cutdeta_, cutminv_, cutdelr_, cutpt_;  // cuts are on=true or off=false

  // typedefs
  typedef std::vector<T1> T1Collection;
  typedef edm::Ref<T1Collection> T1Ref;
  typedef std::vector<T2> T2Collection;
  typedef edm::Ref<T2Collection> T2Ref;
  typedef std::vector<T3> T3Collection;
  typedef edm::Ref<T3Collection> T3Ref;
};

//
// class implementation
//

//
// constructors and destructor
//
template <typename T1, typename T2, typename T3>
HLTDoubletSinglet<T1, T2, T3>::HLTDoubletSinglet(const edm::ParameterSet& iConfig)
    : HLTFilter(iConfig),
      originTag1_(iConfig.getParameter<std::vector<edm::InputTag>>("originTag1")),
      originTag2_(iConfig.getParameter<std::vector<edm::InputTag>>("originTag2")),
      originTag3_(iConfig.getParameter<std::vector<edm::InputTag>>("originTag3")),
      inputTag1_(iConfig.getParameter<edm::InputTag>("inputTag1")),
      inputTag2_(iConfig.getParameter<edm::InputTag>("inputTag2")),
      inputTag3_(iConfig.getParameter<edm::InputTag>("inputTag3")),
      inputToken1_(consumes(inputTag1_)),
      inputToken2_(consumes(inputTag2_)),
      inputToken3_(consumes(inputTag3_)),
      triggerType1_(iConfig.getParameter<int>("triggerType1")),
      triggerType2_(iConfig.getParameter<int>("triggerType2")),
      triggerType3_(iConfig.getParameter<int>("triggerType3")),
      min_Dphi_(iConfig.getParameter<double>("MinDphi")),
      max_Dphi_(iConfig.getParameter<double>("MaxDphi")),
      min_Deta_(iConfig.getParameter<double>("MinDeta")),
      max_Deta_(iConfig.getParameter<double>("MaxDeta")),
      min_Minv_(iConfig.getParameter<double>("MinMinv")),
      max_Minv_(iConfig.getParameter<double>("MaxMinv")),
      min_DelR_(iConfig.getParameter<double>("MinDelR")),
      max_DelR_(iConfig.getParameter<double>("MaxDelR")),
      min_Pt_(iConfig.getParameter<double>("MinPt")),
      max_Pt_(iConfig.getParameter<double>("MaxPt")),
      min_N_(iConfig.getParameter<int>("MinN")),
      same12_(inputTag1_.encode() == inputTag2_.encode()),    // same collections to be compared?
      same13_(inputTag1_.encode() == inputTag3_.encode()),    // same collections to be compared?
      same23_(inputTag2_.encode() == inputTag3_.encode()),    // same collections to be compared?
      min_DelR2_(min_DelR_ < 0 ? 0 : min_DelR_ * min_DelR_),  // avoid computing sqrt(R2)
      max_DelR2_(max_DelR_ < 0 ? 0 : max_DelR_ * max_DelR_),  // avoid computing sqrt(R2)
      cutdphi_(min_Dphi_ <= max_Dphi_),                       // cut active?
      cutdeta_(min_Deta_ <= max_Deta_),                       // cut active?
      cutminv_(min_Minv_ <= max_Minv_),                       // cut active?
      cutdelr_(min_DelR_ <= max_DelR_),                       // cut active?
      cutpt_(min_Pt_ <= max_Pt_)                              // cut active?
{
  LogDebug("") << "InputTags and cuts : " << inputTag1_.encode() << " " << inputTag2_.encode() << " "
               << inputTag3_.encode() << triggerType1_ << " " << triggerType2_ << " " << triggerType3_ << " Dphi ["
               << min_Dphi_ << " " << max_Dphi_ << "]"
               << " Deta [" << min_Deta_ << " " << max_Deta_ << "]"
               << " Minv [" << min_Minv_ << " " << max_Minv_ << "]"
               << " DelR [" << min_DelR_ << " " << max_DelR_ << "]"
               << " Pt   [" << min_Pt_ << " " << max_Pt_ << "]"
               << " MinN =" << min_N_ << " same12/same13/same23/dphi/deta/minv/delr/pt " << same12_ << same13_
               << same23_ << cutdphi_ << cutdeta_ << cutminv_ << cutdelr_ << cutpt_;
  if (cutdelr_ && max_DelR_ <= 0)
    edm::LogWarning("HLTDoubletSinglet")
        << " moduleLabel: " << moduleLabel()
        << "Warning: The deltaR requirement is active, but its range is invalid: DelR [" << min_DelR_ << " "
        << max_DelR_ << "]";
}

template <typename T1, typename T2, typename T3>
HLTDoubletSinglet<T1, T2, T3>::~HLTDoubletSinglet() = default;

template <typename T1, typename T2, typename T3>
void HLTDoubletSinglet<T1, T2, T3>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
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
  desc.add<double>("MinDphi", +1.0);
  desc.add<double>("MaxDphi", -1.0);
  desc.add<double>("MinDeta", +1.0);
  desc.add<double>("MaxDeta", -1.0);
  desc.add<double>("MinMinv", +1.0);
  desc.add<double>("MaxMinv", -1.0);
  desc.add<double>("MinDelR", +1.0);
  desc.add<double>("MaxDelR", -1.0);
  desc.add<double>("MinPt", +1.0);
  desc.add<double>("MaxPt", -1.0);
  desc.add<int>("MinN", 1);
  descriptions.add(defaultModuleLabel<HLTDoubletSinglet<T1, T2, T3>>(), desc);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
template <typename T1, typename T2, typename T3>
bool HLTDoubletSinglet<T1, T2, T3>::hltFilter(edm::Event& iEvent,
                                              const edm::EventSetup& iSetup,
                                              trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;

  // All HLT filters must create and fill an HLT filter object,
  // recording any reconstructed physics objects satisfying (or not)
  // this HLT filter, and place it in the Event.

  int n(0);

  LogVerbatim("HLTDoubletSinglet") << " moduleLabel: " << moduleLabel() << " 0 ";

  // get hold of pre-filtered object collections
  std::vector<T1Ref> coll1;
  auto const& objsWithRefs1 = iEvent.get(inputToken1_);
  objsWithRefs1.getObjects(triggerType1_, coll1);
  std::vector<T2Ref> coll2;
  auto const& objsWithRefs2 = iEvent.get(inputToken2_);
  objsWithRefs2.getObjects(triggerType2_, coll2);
  std::vector<T3Ref> coll3;
  auto const& objsWithRefs3 = iEvent.get(inputToken3_);
  objsWithRefs3.getObjects(triggerType3_, coll3);

  const size_type n1(coll1.size());
  const size_type n2(coll2.size());
  const size_type n3(coll3.size());

  if (saveTags()) {
    InputTag tagOld;
    for (size_t i = 0; i < originTag1_.size(); ++i) {
      filterproduct.addCollectionTag(originTag1_[i]);
      LogVerbatim("HLTDoubletSinglet") << " moduleLabel: " << moduleLabel() << " 1a/" << i << " "
                                       << originTag1_[i].encode();
    }
    tagOld = InputTag();
    for (size_type i1 = 0; i1 != n1; ++i1) {
      const ProductID pid(coll1[i1].id());
      const auto& prov = iEvent.getStableProvenance(pid);
      const string& label(prov.moduleLabel());
      const string& instance(prov.productInstanceName());
      const string& process(prov.processName());
      InputTag tagNew(InputTag(label, instance, process));
      if (tagOld.encode() != tagNew.encode()) {
        filterproduct.addCollectionTag(tagNew);
        tagOld = tagNew;
        LogVerbatim("HLTDoubletSinglet") << " moduleLabel: " << moduleLabel() << " 1b " << tagNew.encode();
      }
    }
    for (size_t i = 0; i < originTag2_.size(); ++i) {
      filterproduct.addCollectionTag(originTag2_[i]);
      LogVerbatim("HLTDoubletSinglet") << " moduleLabel: " << moduleLabel() << " 2a/" << i << " "
                                       << originTag2_[i].encode();
    }
    tagOld = InputTag();
    for (size_type i2 = 0; i2 != n2; ++i2) {
      const ProductID pid(coll2[i2].id());
      const auto& prov = iEvent.getStableProvenance(pid);
      const string& label(prov.moduleLabel());
      const string& instance(prov.productInstanceName());
      const string& process(prov.processName());
      InputTag tagNew(InputTag(label, instance, process));
      if (tagOld.encode() != tagNew.encode()) {
        filterproduct.addCollectionTag(tagNew);
        tagOld = tagNew;
        LogVerbatim("HLTDoubletSinglet") << " moduleLabel: " << moduleLabel() << " 2b " << tagNew.encode();
      }
    }
    for (size_t i = 0; i < originTag3_.size(); ++i) {
      filterproduct.addCollectionTag(originTag3_[i]);
      LogVerbatim("HLTDoubletSinglet") << " moduleLabel: " << moduleLabel() << " 3a/" << i << " "
                                       << originTag3_[i].encode();
    }
    tagOld = InputTag();
    for (size_type i3 = 0; i3 != n3; ++i3) {
      const ProductID pid(coll3[i3].id());
      const auto& prov = iEvent.getStableProvenance(pid);
      const string& label(prov.moduleLabel());
      const string& instance(prov.productInstanceName());
      const string& process(prov.processName());
      InputTag tagNew(InputTag(label, instance, process));
      if (tagOld.encode() != tagNew.encode()) {
        filterproduct.addCollectionTag(tagNew);
        tagOld = tagNew;
        LogVerbatim("HLTDoubletSinglet") << " moduleLabel: " << moduleLabel() << " 3b " << tagNew.encode();
      }
    }

    T1Ref r1;
    T2Ref r2;
    T3Ref r3;
    Candidate::LorentzVector p1, p2, p3, p13, p23;
    for (size_t i1 = 0; i1 != n1; i1++) {
      r1 = coll1[i1];
      p1 = r1->p4();
      auto const i2_min = (same12_ ? i1 + 1 : 0);
      for (size_t i2 = i2_min; i2 != n2; i2++) {
        r2 = coll2[i2];
        p2 = r2->p4();

        auto const i3_min = (same23_ ? i2_min + 1 : (same13_ ? i1 + 1 : 0));
        for (size_t i3 = i3_min; i3 != n3; i3++) {
          r3 = coll3[i3];
          p3 = r3->p4();

          //deltaPhi
          auto const dPhi13(std::abs(deltaPhi(p1.phi(), p3.phi())));
          if (cutdphi_ && (min_Dphi_ > dPhi13 || dPhi13 > max_Dphi_))
            continue;
          auto const dPhi23(std::abs(deltaPhi(p2.phi(), p3.phi())));
          if (cutdphi_ && (min_Dphi_ > dPhi23 || dPhi23 > max_Dphi_))
            continue;

          //deltaEta
          auto const dEta13(std::abs(p1.eta() - p3.eta()));
          if (cutdeta_ && (min_Deta_ > dEta13 || dEta13 > max_Deta_))
            continue;
          auto const dEta23(std::abs(p2.eta() - p3.eta()));
          if (cutdeta_ && (min_Deta_ > dEta23 || dEta23 > max_Deta_))
            continue;

          //deltaR
          auto const delR2_13(dPhi13 * dPhi13 + dEta13 * dEta13);
          if (cutdelr_ && (min_DelR2_ > delR2_13 || delR2_13 > max_DelR2_))
            continue;
          auto const delR2_23(dPhi23 * dPhi23 + dEta23 * dEta23);
          if (cutdelr_ && (min_DelR2_ > delR2_23 || delR2_23 > max_DelR2_))
            continue;

          //Pt and Minv
          p13 = p1 + p3;
          auto const mInv13(std::abs(p13.mass()));
          if (cutminv_ && (min_Minv_ > mInv13 || mInv13 > max_Minv_))
            continue;
          auto const pt13(p13.pt());
          if (cutpt_ && (min_Pt_ > pt13 || pt13 > max_Pt_))
            continue;

          p23 = p2 + p3;
          auto const mInv23(std::abs(p23.mass()));
          if (cutminv_ && (min_Minv_ > mInv23 || mInv23 > max_Minv_))
            continue;
          auto const pt23(p23.pt());
          if (cutpt_ && (min_Pt_ > pt23 || pt23 > max_Pt_))
            continue;

          n++;
          filterproduct.addObject(triggerType1_, r1);
          filterproduct.addObject(triggerType2_, r2);
          filterproduct.addObject(triggerType3_, r3);
        }
      }
    }
  }

  // filter decision
  return (n >= min_N_);
}

#endif  //HLTrigger_HLTfilters_plugins_HLTDoubletSinglet_h
