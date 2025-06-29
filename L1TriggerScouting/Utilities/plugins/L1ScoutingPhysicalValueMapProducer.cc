// system include files
#include <memory>
#include <sstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/allowedValues.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/L1Scouting/interface/OrbitCollection.h"
#include "L1TriggerScouting/Utilities/interface/conversion.h"

#include "CommonTools/Utils/interface/TypedStringObjectMethodCaller.h"

/*
 * Base class
 */
template <typename T>
class L1ScoutingPhysicalValueMapProducer : public edm::stream::EDProducer<> {
public:
  using TOrbitCollection = OrbitCollection<T>;

  L1ScoutingPhysicalValueMapProducer(edm::ParameterSet const &);
  ~L1ScoutingPhysicalValueMapProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::Event &, edm::EventSetup const &) override;

  void putValueMap(edm::Event &, edm::Handle<TOrbitCollection> &, const std::vector<float> &, const std::string &);

  edm::EDGetTokenT<TOrbitCollection> src_;

  static const std::unordered_map<std::string, std::function<float(int)>> func_lookup_;

  std::vector<std::string> labels_;
  std::vector<TypedStringObjectMethodCaller<T, int>> getters_;
  std::vector<std::function<float(int)>> funcs_;
};

template <typename T>
const std::unordered_map<std::string, std::function<float(int)>> L1ScoutingPhysicalValueMapProducer<T>::func_lookup_ = {
    {"ugmt::fPt", l1ScoutingRun3::ugmt::fPt},
    {"ugmt::fEta", l1ScoutingRun3::ugmt::fEta},
    {"ugmt::fPhi", l1ScoutingRun3::ugmt::fPhi},
    {"ugmt::fPtUnconstrained", l1ScoutingRun3::ugmt::fPtUnconstrained},
    {"ugmt::fEtaAtVtx", l1ScoutingRun3::ugmt::fEtaAtVtx},
    {"ugmt::fPhiAtVtx", l1ScoutingRun3::ugmt::fPhiAtVtx},
    {"demux::fEt", l1ScoutingRun3::demux::fEt},
    {"demux::fEta", l1ScoutingRun3::demux::fEta},
    {"demux::fPhi", l1ScoutingRun3::demux::fPhi},
};

template <typename T>
L1ScoutingPhysicalValueMapProducer<T>::L1ScoutingPhysicalValueMapProducer(edm::ParameterSet const &params)
    : src_(consumes(params.getParameter<edm::InputTag>("src"))) {
  auto conversionsPSet = params.getParameter<edm::ParameterSet>("conversions");
  for (const std::string &retname : conversionsPSet.getParameterNamesForType<edm::ParameterSet>()) {
    labels_.emplace_back(retname);
    const auto &conversionPSet = conversionsPSet.getParameter<edm::ParameterSet>(retname);
    const std::string &arg = conversionPSet.getParameter<std::string>("arg");
    getters_.emplace_back(arg);
    const std::string &func_name = conversionPSet.getParameter<std::string>("func");
    auto it = func_lookup_.find(func_name);
    if (it != func_lookup_.end()) {
      funcs_.emplace_back(it->second);
    } else {
      std::stringstream ss;
      for (auto const &func : func_lookup_)
        ss << "\n" << func.first;
      throw cms::Exception("L1ScoutingPhysicalValueMapProducer")
          << "Unrecognised function: " + func_name + "\nAllowed functions are" + ss.str();
    }
    produces<edm::ValueMap<float>>(retname);
  }
}

template <typename T>
void L1ScoutingPhysicalValueMapProducer<T>::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("src");

  edm::ParameterSetDescription conversion;
  conversion.add<std::string>("func")->setComment("function used to convert");
  conversion.add<std::string>("arg")->setComment("attribute of src to be converted");

  edm::ParameterSetDescription conversions;
  conversions.setComment("a parameter set to define all conversions");
  conversions.addNode(
      edm::ParameterWildcard<edm::ParameterSetDescription>("*", edm::RequireZeroOrMore, true, conversion));
  desc.add<edm::ParameterSetDescription>("conversions", conversions);

  descriptions.addWithDefaultLabel(desc);
}

template <typename T>
void L1ScoutingPhysicalValueMapProducer<T>::produce(edm::Event &iEvent, edm::EventSetup const &) {
  edm::Handle<TOrbitCollection> src = iEvent.getHandle(src_);

  unsigned int nobjs = src->size();
  unsigned int nconversions = labels_.size();

  // convert values
  std::vector<std::vector<float>> converted_values(nconversions, std::vector<float>(nobjs));
  for (unsigned int iobj = 0; iobj < nobjs; iobj++) {
    const auto &obj = (*src)[iobj];
    for (unsigned int iconversion = 0; iconversion < nconversions; iconversion++) {
      int value = (getters_[iconversion])(obj);
      converted_values[iconversion][iobj] = (funcs_[iconversion])(value);
    }
  }

  // put to ValueMap
  for (unsigned int iconversion = 0; iconversion < nconversions; iconversion++) {
    putValueMap(iEvent, src, converted_values[iconversion], labels_[iconversion]);
  }
}

template <typename T>
void L1ScoutingPhysicalValueMapProducer<T>::putValueMap(edm::Event &iEvent,
                                                        edm::Handle<TOrbitCollection> &handle,
                                                        const std::vector<float> &values,
                                                        const std::string &label) {
  std::unique_ptr<edm::ValueMap<float>> valuemap(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler filler(*valuemap);
  filler.insert(handle, values.begin(), values.end());
  filler.fill();
  iEvent.put(std::move(valuemap), label);
}

#include "DataFormats/L1Scouting/interface/L1ScoutingMuon.h"
using L1ScoutingMuonPhysicalValueMapProducer = L1ScoutingPhysicalValueMapProducer<l1ScoutingRun3::Muon>;

#include "DataFormats/L1Scouting/interface/L1ScoutingCalo.h"
using L1ScoutingJetPhysicalValueMapProducer = L1ScoutingPhysicalValueMapProducer<l1ScoutingRun3::Jet>;
using L1ScoutingEGammaPhysicalValueMapProducer = L1ScoutingPhysicalValueMapProducer<l1ScoutingRun3::EGamma>;
using L1ScoutingTauPhysicalValueMapProducer = L1ScoutingPhysicalValueMapProducer<l1ScoutingRun3::Tau>;

DEFINE_FWK_MODULE(L1ScoutingMuonPhysicalValueMapProducer);
DEFINE_FWK_MODULE(L1ScoutingJetPhysicalValueMapProducer);
DEFINE_FWK_MODULE(L1ScoutingEGammaPhysicalValueMapProducer);
DEFINE_FWK_MODULE(L1ScoutingTauPhysicalValueMapProducer);
