#ifndef NanoAOD_ObjectPropertyFromIndexMapTableProducer_h
#define NanoAOD_ObjectPropertyFromIndexMapTableProducer_h

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Association.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include <unordered_map>
#include <vector>

template <typename T, typename M>
class ObjectPropertyFromIndexMapTableProducer : public edm::global::EDProducer<> {
public:
  ObjectPropertyFromIndexMapTableProducer(edm::ParameterSet const& params)
      : objName_(params.getParameter<std::string>("objName")),
        branchName_(params.getParameter<std::string>("branchName")),
        doc_(params.getParameter<std::string>("docString")),
        src_(consumes<T>(params.getParameter<edm::InputTag>("src"))),
        mapToken_(consumes<std::unordered_map<int, M>>(params.getParameter<edm::InputTag>("valueMap"))),
        cut_(params.getParameter<std::string>("cut"), true) {
    produces<nanoaod::FlatTable>();
  }

  ~ObjectPropertyFromIndexMapTableProducer() override {}
  //
  // Because I'm not sure if this can be templated, overload instead...
  std::unique_ptr<nanoaod::FlatTable> fillTable(const std::vector<float>& values, const std::string& objName) const {
    auto tab = std::make_unique<nanoaod::FlatTable>(values.size(), objName, false, true);
    tab->addColumn<float>(branchName_, values, doc_);
    return tab;
  }

  // Because I'm not sure if this can be templated, overload instead...
  std::unique_ptr<nanoaod::FlatTable> fillTable(const std::vector<int>& values, const std::string& objName) const {
    auto tab = std::make_unique<nanoaod::FlatTable>(values.size(), objName, false, true);
    tab->addColumn<int>(branchName_, values, doc_);
    return tab;
  }

  void produce(edm::StreamID id, edm::Event& iEvent, const edm::EventSetup& iSetup) const override {
    edm::Handle<T> objs;
    iEvent.getByToken(src_, objs);

    edm::Handle<std::unordered_map<int, M>> valueMap;
    iEvent.getByToken(mapToken_, valueMap);

    std::vector<M> values;
    for (unsigned int i = 0; i < objs->size(); ++i) {
      edm::Ref<T> obj(objs, i);
      if (cut_(*obj)) {
        if (valueMap->find(i) == valueMap->end())
          throw cms::Exception("ObjectPropertyFromIndexMapTableProducer")
              << "No entry in value map for candidate " << i;
        values.emplace_back(valueMap->at(i));
      }
    }

    auto tab = fillTable(values, objName_);
    iEvent.put(std::move(tab));
  }

protected:
  const std::string objName_, branchName_, doc_;
  const edm::EDGetTokenT<T> src_;
  const edm::EDGetTokenT<std::unordered_map<int, M>> mapToken_;
  const StringCutObjectSelector<typename T::value_type> cut_;
};

#endif
