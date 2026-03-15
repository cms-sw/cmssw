#include "PhysicsTools/NanoAOD/interface/SimpleFlatTableProducer.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"

//
// One-to-many: TICLCandidate -> linked Tracksters
// Or SimTICLCandidate --> SimTracksters
//

class TICLCandidateExtraTableProducer : public SimpleFlatTableProducerBase<TICLCandidate, std::vector<TICLCandidate>> {
public:
  using TProd = edm::Ptr<ticl::Trackster>;

  TICLCandidateExtraTableProducer(edm::ParameterSet const& params)
      : SimpleFlatTableProducerBase<TICLCandidate, std::vector<TICLCandidate>>(params) {
    if (params.existsAs<edm::ParameterSet>("collectionVariables")) {
      edm::ParameterSet const& collectionVarsPSet = params.getParameter<edm::ParameterSet>("collectionVariables");
      for (const auto& coltablename : collectionVarsPSet.getParameterNamesForType<edm::ParameterSet>()) {
        const auto& coltablePSet = collectionVarsPSet.getParameter<edm::ParameterSet>(coltablename);

        CollectionVariableTableInfo coltable;
        coltable.name =
            coltablePSet.existsAs<std::string>("name") ? coltablePSet.getParameter<std::string>("name") : coltablename;
        coltable.doc = coltablePSet.getParameter<std::string>("doc");
        coltable.useCount = coltablePSet.getParameter<bool>("useCount");
        coltable.useOffset = coltablePSet.getParameter<bool>("useOffset");

        this->coltables_.push_back(std::move(coltable));
        produces<nanoaod::FlatTable>(coltables_.back().name + "Table");
      }
    }
  }

  void produce(edm::Event& iEvent, const edm::EventSetup&) override {
    const auto& prod = iEvent.getHandle(this->src_);

    const auto& candidates = *prod;
    const size_t table_size = candidates.size();

    auto out = std::make_unique<nanoaod::FlatTable>(table_size, this->name_, /*singleton*/ false, /*extension*/ false);

    unsigned int coltablesize = 0;
    std::vector<unsigned int> counts;
    counts.reserve(table_size);

    std::vector<uint32_t> tracksterKeys;

    for (const auto& cand : candidates) {
      const auto& children = cand.tracksters();
      counts.push_back(children.size());
      coltablesize += children.size();
      for (const auto& t : children) {
        tracksterKeys.push_back(t.key());
      }
    }

    for (const auto& coltable : this->coltables_) {
      if (coltable.useCount) {
        out->addColumn<uint16_t>("n" + coltable.name, counts, "Count for " + coltable.name);
      }
      if (coltable.useOffset) {
        std::vector<unsigned int> offsets;
        offsets.reserve(counts.size());
        unsigned int offset = 0;
        for (auto c : counts) {
          offsets.push_back(offset);
          offset += c;
        }
        out->addColumn<uint16_t>("o" + coltable.name, offsets, "Offset for " + coltable.name);
      }

      auto outcoltable = std::make_unique<nanoaod::FlatTable>(coltablesize, coltable.name, false, false);

      outcoltable->addColumn<uint32_t>("tracksterIndex", tracksterKeys, "Index of associated Trackster");

      outcoltable->setDoc(coltable.doc);
      iEvent.put(std::move(outcoltable), coltable.name + "Table");
    }

    if (out->nColumns() > 0) {
      out->setDoc(this->doc_);
      iEvent.put(std::move(out));
    }
  }

  std::unique_ptr<nanoaod::FlatTable> fillTable(const edm::Event&,
                                                const edm::Handle<std::vector<TICLCandidate>>&) const override {
    return std::make_unique<nanoaod::FlatTable>();
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc =
        SimpleFlatTableProducerBase<TICLCandidate, std::vector<TICLCandidate>>::baseDescriptions();

    edm::ParameterSetDescription coltable;
    coltable.add<std::string>("name", "hltTiclCandidate");
    coltable.add<std::string>("doc", "TICL Candidates");
    coltable.add<bool>("useCount", true);
    coltable.add<bool>("useOffset", false);
    edm::ParameterSetDescription colvariables;  // unused here
    coltable.add<edm::ParameterSetDescription>("variables", colvariables);

    edm::ParameterSetDescription coltables;
    coltables.addOptionalNode(
        edm::ParameterWildcard<edm::ParameterSetDescription>("*", edm::RequireZeroOrMore, true, coltable), false);

    desc.addOptional<edm::ParameterSetDescription>("collectionVariables", coltables);
    descriptions.addWithDefaultLabel(desc);
  }

protected:
  struct CollectionVariableTableInfo {
    std::string name;
    std::string doc;
    bool useCount;
    bool useOffset;
  };
  std::vector<CollectionVariableTableInfo> coltables_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TICLCandidateExtraTableProducer);
