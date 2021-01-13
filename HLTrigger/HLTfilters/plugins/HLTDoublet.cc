/** \class HLTDoublet
 *
 * See header file for documentation
 *
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTDoublet.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "DataFormats/Candidate/interface/Particle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"
#include <cmath>

//
// constructors and destructor
//
template <typename T1, typename T2>
HLTDoublet<T1, T2>::HLTDoublet(const edm::ParameterSet& iConfig)
    : HLTFilter(iConfig),
      originTag1_(iConfig.template getParameter<std::vector<edm::InputTag>>("originTag1")),
      originTag2_(iConfig.template getParameter<std::vector<edm::InputTag>>("originTag2")),
      inputTag1_(iConfig.template getParameter<edm::InputTag>("inputTag1")),
      inputTag2_(iConfig.template getParameter<edm::InputTag>("inputTag2")),
      inputToken1_(consumes<trigger::TriggerFilterObjectWithRefs>(inputTag1_)),
      inputToken2_(consumes<trigger::TriggerFilterObjectWithRefs>(inputTag2_)),
      triggerType1_(iConfig.template getParameter<int>("triggerType1")),
      triggerType2_(iConfig.template getParameter<int>("triggerType2")),
      min_Dphi_(iConfig.template getParameter<double>("MinDphi")),
      max_Dphi_(iConfig.template getParameter<double>("MaxDphi")),
      min_Deta_(iConfig.template getParameter<double>("MinDeta")),
      max_Deta_(iConfig.template getParameter<double>("MaxDeta")),
      min_Minv_(iConfig.template getParameter<double>("MinMinv")),
      max_Minv_(iConfig.template getParameter<double>("MaxMinv")),
      min_DelR_(iConfig.template getParameter<double>("MinDelR")),
      max_DelR_(iConfig.template getParameter<double>("MaxDelR")),
      min_Pt_(iConfig.template getParameter<double>("MinPt")),
      max_Pt_(iConfig.template getParameter<double>("MaxPt")),
      min_N_(iConfig.template getParameter<int>("MinN")),
      same_(inputTag1_.encode() == inputTag2_.encode()),  // same collections to be compared?
      cutdphi_(min_Dphi_ <= max_Dphi_),                   // cut active?
      cutdeta_(min_Deta_ <= max_Deta_),                   // cut active?
      cutminv_(min_Minv_ <= max_Minv_),                   // cut active?
      cutdelr_(min_DelR_ <= max_DelR_),                   // cut active?
      cutpt_(min_Pt_ <= max_Pt_)                          // cut active?
{
  LogDebug("") << "InputTags and cuts : " << inputTag1_.encode() << " " << inputTag2_.encode() << triggerType1_ << " "
               << triggerType2_ << " Dphi [" << min_Dphi_ << " " << max_Dphi_ << "]"
               << " Deta [" << min_Deta_ << " " << max_Deta_ << "]"
               << " Minv [" << min_Minv_ << " " << max_Minv_ << "]"
               << " DelR [" << min_DelR_ << " " << max_DelR_ << "]"
               << " Pt   [" << min_Pt_ << " " << max_Pt_ << "]"
               << " MinN =" << min_N_ << " same/dphi/deta/minv/delr/pt " << same_ << cutdphi_ << cutdeta_ << cutminv_
               << cutdelr_ << cutpt_;
}

template <typename T1, typename T2>
HLTDoublet<T1, T2>::~HLTDoublet() = default;
template <typename T1, typename T2>
void HLTDoublet<T1, T2>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  std::vector<edm::InputTag> originTag1(1, edm::InputTag("hltOriginal1"));
  std::vector<edm::InputTag> originTag2(1, edm::InputTag("hltOriginal2"));
  desc.add<std::vector<edm::InputTag>>("originTag1", originTag1);
  desc.add<std::vector<edm::InputTag>>("originTag2", originTag2);
  desc.add<edm::InputTag>("inputTag1", edm::InputTag("hltFiltered1"));
  desc.add<edm::InputTag>("inputTag2", edm::InputTag("hltFiltered22"));
  desc.add<int>("triggerType1", 0);
  desc.add<int>("triggerType2", 0);
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
  descriptions.add(defaultModuleLabel<HLTDoublet<T1, T2>>(), desc);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
template <typename T1, typename T2>
bool HLTDoublet<T1, T2>::hltFilter(edm::Event& iEvent,
                                   const edm::EventSetup& iSetup,
                                   trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;

  // All HLT filters must create and fill an HLT filter object,
  // recording any reconstructed physics objects satisfying (or not)
  // this HLT filter, and place it in the Event.

  bool accept(false);

  LogVerbatim("HLTDoublet") << " XXX " << moduleLabel() << " 0 " << std::endl;

  std::vector<T1Ref> coll1;
  std::vector<T2Ref> coll2;

  // get hold of pre-filtered object collections
  Handle<TriggerFilterObjectWithRefs> handle1, handle2;
  if (iEvent.getByToken(inputToken1_, handle1) && iEvent.getByToken(inputToken2_, handle2)) {
    handle1->getObjects(triggerType1_, coll1);
    handle2->getObjects(triggerType2_, coll2);
    const size_type n1(coll1.size());
    const size_type n2(coll2.size());

    if (saveTags()) {
      InputTag tagOld;
      for (unsigned int i = 0; i < originTag1_.size(); ++i) {
        filterproduct.addCollectionTag(originTag1_[i]);
        LogVerbatim("HLTDoublet") << " XXX " << moduleLabel() << " 1a/" << i << " " << originTag1_[i].encode()
                                  << std::endl;
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
          LogVerbatim("HLTDoublet") << " XXX " << moduleLabel() << " 1b " << tagNew.encode() << std::endl;
        }
      }
      for (unsigned int i = 0; i < originTag2_.size(); ++i) {
        filterproduct.addCollectionTag(originTag2_[i]);
        LogVerbatim("HLTDoublet") << " XXX " << moduleLabel() << " 2a/" << i << " " << originTag2_[i].encode()
                                  << std::endl;
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
          LogVerbatim("HLTDoublet") << " XXX " << moduleLabel() << " 2b " << tagNew.encode() << std::endl;
        }
      }
    }

    int n(0);
    T1Ref r1;
    T2Ref r2;
    Particle::LorentzVector p1, p2, p;
    for (unsigned int i1 = 0; i1 != n1; i1++) {
      r1 = coll1[i1];
      p1 = r1->p4();
      unsigned int I(0);
      if (same_) {
        I = i1 + 1;
      }
      for (unsigned int i2 = I; i2 != n2; i2++) {
        r2 = coll2[i2];
        p2 = r2->p4();

        double Dphi(std::abs(p1.phi() - p2.phi()));
        if (Dphi > M_PI)
          Dphi = 2.0 * M_PI - Dphi;

        double Deta(std::abs(p1.eta() - p2.eta()));

        p = p1 + p2;
        double Minv(std::abs(p.mass()));
        double Pt(p.pt());

        double DelR(sqrt(Dphi * Dphi + Deta * Deta));

        if (((!cutdphi_) || ((min_Dphi_ <= Dphi) && (Dphi <= max_Dphi_))) &&
            ((!cutdeta_) || ((min_Deta_ <= Deta) && (Deta <= max_Deta_))) &&
            ((!cutminv_) || ((min_Minv_ <= Minv) && (Minv <= max_Minv_))) &&
            ((!cutdelr_) || ((min_DelR_ <= DelR) && (DelR <= max_DelR_))) &&
            ((!cutpt_) || ((min_Pt_ <= Pt) && (Pt <= max_Pt_)))) {
          n++;
          filterproduct.addObject(triggerType1_, r1);
          filterproduct.addObject(triggerType2_, r2);
        }
      }
    }
    // filter decision
    accept = (n >= min_N_);
  }

  return accept;
}
