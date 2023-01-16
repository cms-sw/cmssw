// system include files
#include <memory>
#include <sstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"

class TriggerObjectTableProducer : public edm::stream::EDProducer<> {
public:
  explicit TriggerObjectTableProducer(const edm::ParameterSet &iConfig)
      : name_(iConfig.getParameter<std::string>("name")),
        src_(consumes<std::vector<pat::TriggerObjectStandAlone>>(iConfig.getParameter<edm::InputTag>("src"))),
        l1EG_(consumes<l1t::EGammaBxCollection>(iConfig.getParameter<edm::InputTag>("l1EG"))),
        l1Sum_(consumes<l1t::EtSumBxCollection>(iConfig.getParameter<edm::InputTag>("l1Sum"))),
        l1Jet_(consumes<l1t::JetBxCollection>(iConfig.getParameter<edm::InputTag>("l1Jet"))),
        l1Muon_(consumes<l1t::MuonBxCollection>(iConfig.getParameter<edm::InputTag>("l1Muon"))),
        l1Tau_(consumes<l1t::TauBxCollection>(iConfig.getParameter<edm::InputTag>("l1Tau"))) {
    edm::ParameterSet selPSet = iConfig.getParameter<edm::ParameterSet>("selections");
    const auto selNames = selPSet.getParameterNames();
    std::stringstream idstr, qualitystr;
    idstr << "ID of the object: ";
    for (const auto &name : selNames) {
      sels_.emplace_back(selPSet.getParameter<edm::ParameterSet>(name));
      const auto &sel = sels_.back();
      idstr << sel.id << " = " << name + sel.doc;
      if (sels_.size() < selNames.size())
        idstr << ", ";
      if (!sel.qualityBitsDoc.empty()) {
        qualitystr << sel.qualityBitsDoc << " for " << name << "; ";
      }
    }
    idDoc_ = idstr.str();
    bitsDoc_ = qualitystr.str();

    produces<nanoaod::FlatTable>();
  }

  ~TriggerObjectTableProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::Event &, edm::EventSetup const &) override;

  std::string name_;
  edm::EDGetTokenT<std::vector<pat::TriggerObjectStandAlone>> src_;
  std::string idDoc_, bitsDoc_;

  edm::EDGetTokenT<l1t::EGammaBxCollection> l1EG_;
  edm::EDGetTokenT<l1t::EtSumBxCollection> l1Sum_;
  edm::EDGetTokenT<l1t::JetBxCollection> l1Jet_;
  edm::EDGetTokenT<l1t::MuonBxCollection> l1Muon_;
  edm::EDGetTokenT<l1t::TauBxCollection> l1Tau_;

  struct SelectedObject {
    std::string doc;
    int id;
    StringCutObjectSelector<pat::TriggerObjectStandAlone> cut;
    StringCutObjectSelector<pat::TriggerObjectStandAlone> l1cut, l1cut_2, l2cut;
    float l1DR2, l1DR2_2, l2DR2;
    bool skipObjectsNotPassingQualityBits;
    StringObjectFunction<pat::TriggerObjectStandAlone> qualityBits;
    std::string qualityBitsDoc;

    SelectedObject(const edm::ParameterSet &pset)
        : doc(pset.getParameter<std::string>("doc")),
          id(pset.getParameter<int>("id")),
          cut(pset.getParameter<std::string>("sel")),
          l1cut(""),
          l1cut_2(""),
          l2cut(""),
          l1DR2(-1),
          l1DR2_2(-1),
          l2DR2(-1),
          skipObjectsNotPassingQualityBits(pset.getParameter<bool>("skipObjectsNotPassingQualityBits")),
          qualityBits("0"),   //will be overwritten from configuration
          qualityBitsDoc("")  //will be created from configuration
    {
      if (!doc.empty()) {
        doc = "(" + doc + ")";
      }
      std::vector<edm::ParameterSet> qualityBitsConfig =
          pset.getParameter<std::vector<edm::ParameterSet>>("qualityBits");
      std::stringstream qualityBitsFunc;
      std::vector<bool> bits(qualityBitsConfig.size(), false);
      for (size_t i = 0; i != qualityBitsConfig.size(); ++i) {
        if (i != 0) {
          qualityBitsFunc << " + ";
          qualityBitsDoc += ", ";
        }
        unsigned int bit = i;
        if (qualityBitsConfig[i].existsAs<unsigned int>("bit"))
          bit = qualityBitsConfig[i].getParameter<unsigned int>("bit");
        assert(!bits[bit] && "a quality bit was inserted twice");  // the bit should not have been set already
        assert(bit < 31 && "quality bits are store on 32 bit");
        bits[bit] = true;
        qualityBitsFunc << std::to_string(int(pow(2, bit))) << "*("
                        << qualityBitsConfig[i].getParameter<std::string>("selection") << ")";
        qualityBitsDoc += std::to_string(bit) + " => " + qualityBitsConfig[i].getParameter<std::string>("doc");
      }
      if (!qualityBitsFunc.str().empty()) {
        //std::cout << "The quality bit string is :" << qualityBitsFunc.str() << std::endl;
        //std::cout << "The quality bit documentation is :" << qualityBitsDoc << std::endl;
        qualityBits = StringObjectFunction<pat::TriggerObjectStandAlone>(qualityBitsFunc.str());
      }
      if (pset.existsAs<std::string>("l1seed")) {
        l1cut = StringCutObjectSelector<pat::TriggerObjectStandAlone>(pset.getParameter<std::string>("l1seed"));
        l1DR2 = std::pow(pset.getParameter<double>("l1deltaR"), 2);
      }
      if (pset.existsAs<std::string>("l1seed_2")) {
        l1cut_2 = StringCutObjectSelector<pat::TriggerObjectStandAlone>(pset.getParameter<std::string>("l1seed_2"));
        l1DR2_2 = std::pow(pset.getParameter<double>("l1deltaR_2"), 2);
      }
      if (pset.existsAs<std::string>("l2seed")) {
        l2cut = StringCutObjectSelector<pat::TriggerObjectStandAlone>(pset.getParameter<std::string>("l2seed"));
        l2DR2 = std::pow(pset.getParameter<double>("l2deltaR"), 2);
      }
    }

    bool match(const pat::TriggerObjectStandAlone &obj) const { return cut(obj); }
  };

  std::vector<SelectedObject> sels_;
};

// ------------ method called to produce the data  ------------
void TriggerObjectTableProducer::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  const auto &trigObjs = iEvent.get(src_);

  std::vector<std::pair<const pat::TriggerObjectStandAlone *, const SelectedObject *>> selected;
  std::map<int, std::map<const pat::TriggerObjectStandAlone *, int>> selected_bits;
  for (const auto &obj : trigObjs) {
    for (const auto &sel : sels_) {
      if (sel.match(obj)) {
        selected_bits[sel.id][&obj] = int(sel.qualityBits(obj));
        if (sel.skipObjectsNotPassingQualityBits ? (selected_bits[sel.id][&obj] > 0) : true) {
          selected.emplace_back(&obj, &sel);
        }
      }
    }
  }

  // Self-cleaning
  for (unsigned int i = 0; i < selected.size(); ++i) {
    const auto &obj = *selected[i].first;
    const auto &sel = *selected[i].second;

    for (unsigned int j = 0; j < i; ++j) {
      const auto &obj2 = *selected[j].first;
      const auto &sel2 = *selected[j].second;
      if (sel.id == sel2.id && abs(obj.pt() - obj2.pt()) < 1e-6 && deltaR2(obj, obj2) < 1e-6) {
        selected_bits[sel.id][&obj2] |= selected_bits[sel.id][&obj];  //Keep filters from all the objects
        selected.erase(selected.begin() + i);
        i--;
      }
    }
  }

  const auto &l1EG = iEvent.get(l1EG_);
  const auto &l1Sum = iEvent.get(l1Sum_);
  const auto &l1Jet = iEvent.get(l1Jet_);
  const auto &l1Muon = iEvent.get(l1Muon_);
  const auto &l1Tau = iEvent.get(l1Tau_);

  std::vector<pair<pat::TriggerObjectStandAlone, int>> l1Objects;
  l1Objects.reserve(l1EG.size(0) + l1Sum.size(0) + l1Jet.size(0) + l1Muon.size(0) + l1Tau.size(0));

  // no range-based for because we want bx=0 only
  for (l1t::EGammaBxCollection::const_iterator it = l1EG.begin(0); it != l1EG.end(0); it++) {
    pat::TriggerObjectStandAlone l1obj(it->p4());
    l1obj.setCollection("L1EG");
    l1obj.addTriggerObjectType(trigger::TriggerL1EG);
    l1Objects.emplace_back(l1obj, it->hwIso());
  }

  for (l1t::EtSumBxCollection::const_iterator it = l1Sum.begin(0); it != l1Sum.end(0); it++) {
    pat::TriggerObjectStandAlone l1obj(it->p4());

    switch (it->getType()) {
      case l1t::EtSum::EtSumType::kMissingEt:
        l1obj.addTriggerObjectType(trigger::TriggerL1ETM);
        l1obj.setCollection("L1ETM");
        break;

      case l1t::EtSum::EtSumType::kMissingEtHF:
        l1obj.addTriggerObjectType(trigger::TriggerL1ETM);
        l1obj.setCollection("L1ETMHF");
        break;

      case l1t::EtSum::EtSumType::kTotalEt:
        l1obj.addTriggerObjectType(trigger::TriggerL1ETT);
        l1obj.setCollection("L1ETT");
        break;

      case l1t::EtSum::EtSumType::kTotalEtEm:
        l1obj.addTriggerObjectType(trigger::TriggerL1ETT);
        l1obj.setCollection("L1ETEm");
        break;

      case l1t::EtSum::EtSumType::kTotalHt:
        l1obj.addTriggerObjectType(trigger::TriggerL1HTT);
        l1obj.setCollection("L1HTT");
        break;

      case l1t::EtSum::EtSumType::kTotalHtHF:
        l1obj.addTriggerObjectType(trigger::TriggerL1HTT);
        l1obj.setCollection("L1HTTHF");
        break;

      case l1t::EtSum::EtSumType::kMissingHt:
        l1obj.addTriggerObjectType(trigger::TriggerL1HTM);
        l1obj.setCollection("L1HTM");
        break;

      case l1t::EtSum::EtSumType::kMissingHtHF:
        l1obj.addTriggerObjectType(trigger::TriggerL1HTM);
        l1obj.setCollection("L1HTMHF");
        break;

      default:
        continue;
    }

    l1Objects.emplace_back(l1obj, it->hwIso());
  }

  for (l1t::JetBxCollection::const_iterator it = l1Jet.begin(0); it != l1Jet.end(0); it++) {
    pat::TriggerObjectStandAlone l1obj(it->p4());
    l1obj.setCollection("L1Jet");
    l1obj.addTriggerObjectType(trigger::TriggerL1Jet);
    l1Objects.emplace_back(l1obj, it->hwIso());
  }

  for (l1t::MuonBxCollection::const_iterator it = l1Muon.begin(0); it != l1Muon.end(0); it++) {
    pat::TriggerObjectStandAlone l1obj(it->p4());
    l1obj.setCollection("L1Mu");
    l1obj.addTriggerObjectType(trigger::TriggerL1Mu);
    l1obj.setCharge(it->charge());
    l1Objects.emplace_back(l1obj, it->hwIso());
  }

  for (l1t::TauBxCollection::const_iterator it = l1Tau.begin(0); it != l1Tau.end(0); it++) {
    pat::TriggerObjectStandAlone l1obj(it->p4());
    l1obj.setCollection("L1Tau");
    l1obj.addTriggerObjectType(trigger::TriggerL1Tau);
    l1Objects.emplace_back(l1obj, it->hwIso());
  }

  unsigned int nobj = selected.size();
  std::vector<float> pt(nobj, 0), eta(nobj, 0), phi(nobj, 0), l1pt(nobj, 0), l1pt_2(nobj, 0), l2pt(nobj, 0);
  std::vector<int16_t> l1charge(nobj, 0);
  std::vector<uint16_t> id(nobj, 0);
  std::vector<int> bits(nobj, 0), l1iso(nobj, 0);
  for (unsigned int i = 0; i < nobj; ++i) {
    const auto &obj = *selected[i].first;
    const auto &sel = *selected[i].second;
    pt[i] = obj.pt();
    eta[i] = obj.eta();
    phi[i] = obj.phi();
    id[i] = sel.id;
    bits[i] = selected_bits[sel.id][&obj];
    if (sel.l1DR2 > 0) {
      float best = sel.l1DR2;
      for (const auto &l1obj : l1Objects) {
        const auto &seed = l1obj.first;
        float dr2 = deltaR2(seed, obj);
        if (dr2 < best && sel.l1cut(seed)) {
          best = dr2;
          l1pt[i] = seed.pt();
          l1iso[i] = l1obj.second;
          l1charge[i] = seed.charge();
        }
      }
    }
    if (sel.l1DR2_2 > 0) {
      float best = sel.l1DR2_2;
      for (const auto &l1obj : l1Objects) {
        const auto &seed = l1obj.first;
        float dr2 = deltaR2(seed, obj);
        if (dr2 < best && sel.l1cut_2(seed)) {
          best = dr2;
          l1pt_2[i] = seed.pt();
        }
      }
    }
    if (sel.l2DR2 > 0) {
      float best = sel.l2DR2;
      for (const auto &seed : trigObjs) {
        float dr2 = deltaR2(seed, obj);
        if (dr2 < best && sel.l2cut(seed)) {
          best = dr2;
          l2pt[i] = seed.pt();
        }
      }
    }
  }

  auto tab = std::make_unique<nanoaod::FlatTable>(nobj, name_, false, false);
  tab->addColumn<uint16_t>("id", id, idDoc_);
  tab->addColumn<float>("pt", pt, "pt", 12);
  tab->addColumn<float>("eta", eta, "eta", 12);
  tab->addColumn<float>("phi", phi, "phi", 12);
  tab->addColumn<float>("l1pt", l1pt, "pt of associated L1 seed", 8);
  tab->addColumn<int>("l1iso", l1iso, "iso of associated L1 seed");
  tab->addColumn<int16_t>("l1charge", l1charge, "charge of associated L1 seed");
  tab->addColumn<float>("l1pt_2", l1pt_2, "pt of associated secondary L1 seed", 8);
  tab->addColumn<float>("l2pt", l2pt, "pt of associated 'L2' seed (i.e. HLT before tracking/PF)", 10);
  tab->addColumn<int>("filterBits", bits, "extra bits of associated information: " + bitsDoc_);
  iEvent.put(std::move(tab));
}

void TriggerObjectTableProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("name")->setComment("name of the flat table output");
  desc.add<edm::InputTag>("src")->setComment("pat::TriggerObjectStandAlone input collection");
  desc.add<edm::InputTag>("l1EG")->setComment("l1t::EGammaBxCollection input collection");
  desc.add<edm::InputTag>("l1Sum")->setComment("l1t::EtSumBxCollection input collection");
  desc.add<edm::InputTag>("l1Jet")->setComment("l1t::JetBxCollection input collection");
  desc.add<edm::InputTag>("l1Muon")->setComment("l1t::MuonBxCollection input collection");
  desc.add<edm::InputTag>("l1Tau")->setComment("l1t::TauBxCollection input collection");

  edm::ParameterSetDescription selection;
  selection.setComment("a parameterset to define a trigger collection in flat table");
  selection.add<std::string>("doc", "")->setComment(
      "optional additional info to be added to the table doc for that object");
  selection.add<int>("id")->setComment("identifier of the trigger collection in the flat table");
  selection.add<std::string>("sel")->setComment("function to selection on pat::TriggerObjectStandAlone");
  selection.add<bool>("skipObjectsNotPassingQualityBits")->setComment("flag to skip object on quality bit");

  edm::ParameterSetDescription bit;
  bit.add<std::string>("selection")->setComment("function on pat::TriggerObjectStandAlone to define quality bit");
  bit.add<std::string>("doc")->setComment("definition of the quality bit");
  bit.addOptional<uint>("bit")->setComment("value of the bit, if not the order in the VPset");
  bit.setComment("parameter set to define quality bit of matching object");
  selection.addVPSet("qualityBits", bit);

  selection.ifExists(edm::ParameterDescription<std::string>("l1seed", "selection on pat::TriggerObjectStandAlone"),
                     edm::ParameterDescription<double>(
                         "l1deltaR", "deltaR criteria to match pat::TriggerObjectStandAlone to L1 primitive"));
  selection.ifExists(edm::ParameterDescription<std::string>("l1seed_2", "selection on pat::TriggerObjectStandAlone"),
                     edm::ParameterDescription<double>(
                         "l1deltaR_2", "deltaR criteria to match pat::TriggerObjectStandAlone to L1 primitive"));
  selection.ifExists(edm::ParameterDescription<std::string>("l2seed", "selection on pat::TriggerObjectStandAlone"),
                     edm::ParameterDescription<double>(
                         "l2deltaR", "deltaR criteria to match pat::TriggerObjectStandAlone to 'L2' primitive"));

  edm::ParameterWildcard<edm::ParameterSetDescription> selectionsNode("*", edm::RequireAtLeastOne, true, selection);
  edm::ParameterSetDescription selections;
  selections.addNode(selectionsNode);
  desc.add<edm::ParameterSetDescription>("selections", selections);

  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TriggerObjectTableProducer);
