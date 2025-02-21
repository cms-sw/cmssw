#include <numeric>  // iota
#include "PhysicsTools/NanoAOD/interface/SimpleFlatTableProducer.h"

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"

template <typename ObjType, typename ValType>
class DummyVariable : public Variable<ObjType> {
public:
  DummyVariable(const std::string &aname, const edm::ParameterSet &cfg) : Variable<ObjType>(aname, cfg) {}
  ~DummyVariable() override {}

  void fill(std::vector<const ObjType *> &selobjs, nanoaod::FlatTable &out) const override {
    /**
    std::vector<ValType> vals(selobjs.size());
    for (unsigned int i = 0, n = vals.size(); i < n; ++i) {
      vals[i] = func_(*selobjs[i]);
      if constexpr (std::is_same<ValType, float>()) {
        if (this->precision_ == -2) {
          auto prec = precisionFunc_(*selobjs[i]);
          if (prec > 0) {
            vals[i] = MiniFloatConverter::reduceMantissaToNbitsRounding(vals[i], prec);
          }
        }
      }
    }
    out.template addColumn<ValType>(this->name_, vals, this->doc_, this->precision_);
    */
  }
};

template <typename ObjType, typename ValType>
class DummyCollectionVariable : public CollectionVariable<ObjType> {
public:
  DummyCollectionVariable(const std::string &aname, const edm::ParameterSet &cfg)
      : CollectionVariable<ObjType>(aname, cfg) {}
  ~DummyCollectionVariable() override {}

  std::unique_ptr<std::vector<unsigned int>> getCounts(std::vector<const ObjType *> &selobjs) const override {
    auto counts = std::make_unique<std::vector<unsigned int>>();
    return counts;
    /**
    for (auto const &obj : selobjs)
      counts->push_back(func_(*obj).size());
    return counts;
    */
  }

  void fill(std::vector<const ObjType *> &selobjs, nanoaod::FlatTable &out) const override {
    /**
    std::vector<ValType> vals;
    for (unsigned int i = 0; i < selobjs.size(); ++i) {
      for (ValType val : func_(*selobjs[i])) {
        if constexpr (std::is_same<ValType, float>()) {
          if (this->precision_ == -2) {
            auto prec = precisionFunc_(*selobjs[i]);
            if (prec > 0) {
              val = MiniFloatConverter::reduceMantissaToNbitsRounding(val, prec);
            }
          }
        }
        vals.push_back(val);
      }
    }
    out.template addColumn<ValType>(this->name_, vals, this->doc_, this->precision_);
    */
  }
};

// TODO: specialize only for the required association types.
template <typename T>
class SimpleAssociationCollectionFlatTableProducer : public SimpleFlatTableProducerBase<T, T> {
public:
  SimpleAssociationCollectionFlatTableProducer(edm::ParameterSet const &params)
      : SimpleFlatTableProducerBase<T, T>(params) {
    if (params.existsAs<edm::ParameterSet>("collectionVariables")) {
      edm::ParameterSet const &collectionVarsPSet = params.getParameter<edm::ParameterSet>("collectionVariables");
      for (const std::string &coltablename :
           collectionVarsPSet.getParameterNamesForType<edm::ParameterSet>()) {  // tables of variables
        const auto &coltablePSet = collectionVarsPSet.getParameter<edm::ParameterSet>(coltablename);
        CollectionVariableTableInfo coltable;
        coltable.name =
            coltablePSet.existsAs<std::string>("name") ? coltablePSet.getParameter<std::string>("name") : coltablename;
        coltable.doc = coltablePSet.getParameter<std::string>("doc");
        coltable.useCount = coltablePSet.getParameter<bool>("useCount");
        coltable.useOffset = coltablePSet.getParameter<bool>("useOffset");
        const auto &colvarsPSet = coltablePSet.getParameter<edm::ParameterSet>("variables");
        for (const std::string &colvarname : colvarsPSet.getParameterNamesForType<edm::ParameterSet>()) {  // variables
          const auto &colvarPSet = colvarsPSet.getParameter<edm::ParameterSet>(colvarname);
          const std::string &type = colvarPSet.getParameter<std::string>("type");
          if (type == "int")
            coltable.colvars.push_back(std::make_unique<IntVectorVar>(colvarname, colvarPSet));
          else if (type == "uint")
            coltable.colvars.push_back(std::make_unique<UIntVectorVar>(colvarname, colvarPSet));
          else if (type == "float")
            coltable.colvars.push_back(std::make_unique<FloatVectorVar>(colvarname, colvarPSet));
          else if (type == "double")
            coltable.colvars.push_back(std::make_unique<DoubleVectorVar>(colvarname, colvarPSet));
          else if (type == "uint8")
            coltable.colvars.push_back(std::make_unique<UInt8VectorVar>(colvarname, colvarPSet));
          else if (type == "int16")
            coltable.colvars.push_back(std::make_unique<Int16VectorVar>(colvarname, colvarPSet));
          else if (type == "uint16")
            coltable.colvars.push_back(std::make_unique<UInt16VectorVar>(colvarname, colvarPSet));
          else
            throw cms::Exception("Configuration",
                                 "unsupported type " + type + " for variable " + colvarname + " in " + coltablename);
        }
        this->coltables.push_back(std::move(coltable));
        edm::stream::EDProducer<>::produces<nanoaod::FlatTable>(coltables.back().name + "Table");
      }
    }
  }

  ~SimpleAssociationCollectionFlatTableProducer() override {}

  void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) override {
    edm::Handle<T> prod;
    iEvent.getByToken(SimpleFlatTableProducerBase<T, T>::src_, prod);

    //    std::unique_ptr<nanoaod::FlatTable> out = fillTable(iEvent, src);
    auto table_size = prod->getMap().size();
    std::cout << "MR creating table of size: " << table_size << std::endl;
    auto out = std::make_unique<nanoaod::FlatTable>(table_size, this->name_, false);
    std::vector<int> vals(table_size);
    std::iota(vals.begin(), vals.end(), 0);
    out->template addColumn<int32_t>("srcIdx", vals, "Index of the source objects.");

    // Now proceed with the variable-sized linked objects.
    unsigned int coltablesize = 0;
    std::vector<unsigned int> counts;
    counts.reserve(table_size);
    for (auto const &links : prod->getMap()) {
      counts.push_back(links.size());
      coltablesize += counts.back();
    }
    // add count branch if requested
    out->template addColumn<uint16_t>("n" + coltables[0].name, counts, "counts for linked objects.");
    // add offset branch if requested
    unsigned int offset = 0;
    std::vector<unsigned int> offsets;
    offsets.reserve(table_size);
    for (auto const &count : counts) {
      offsets.push_back(offset);
      offset += count;
    }
    out->template addColumn<uint16_t>("o" + coltables[0].name, offsets, "offsets for linked objects.");

    out->setDoc(SimpleFlatTableProducerBase<T, T>::doc_);
    iEvent.put(std::move(out));

    std::vector<int32_t> all_links;
    all_links.reserve(coltablesize);
    std::vector<float> all_scores;
    all_scores.reserve(coltablesize);
    for (auto const &links : prod->getMap()) {
      for (auto const &alink : links) {
        all_links.push_back(alink.index());
        all_scores.push_back(alink.score());
      }
    }
    std::unique_ptr<nanoaod::FlatTable> outcoltable =
        std::make_unique<nanoaod::FlatTable>(coltablesize, coltables[0].name, false, false);
    outcoltable->template addColumn<int32_t>("indices", all_links, "Indices of linked objects.");

    outcoltable->template addColumn<float>("scores", all_scores, "Scores of linked objects.");
    iEvent.put(std::move(outcoltable), coltables[0].name + "Table");
  }

  // Make compiler happy.
  std::unique_ptr<nanoaod::FlatTable> fillTable(const edm::Event &iEvent, const edm::Handle<T> &prod) const override {
    // Do nothing here and move all the heavy lifting in the produce method.
    auto out = std::make_unique<nanoaod::FlatTable>();
    return out;
  }

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc = SimpleFlatTableProducer<T>::baseDescriptions();
    edm::ParameterSetDescription colvariable;
    colvariable.add<std::string>("expr")->setComment(
        "a function to define the content of the branch in the flat table");
    colvariable.add<std::string>("doc")->setComment("few words of self documentation");
    colvariable.addUntracked<bool>("lazyEval", false)
        ->setComment("if true, can use methods of inheriting classes in `expr`. Can cause problems with threading.");
    colvariable.ifValue(edm::ParameterDescription<std::string>(
                            "type", "int", true, edm::Comment("the c++ type of the branch in the flat table")),
                        edm::allowedValues<std::string>("int", "uint", "float", "double", "uint8", "int16", "uint16"));
    colvariable.addOptionalNode(
        edm::ParameterDescription<int>(
            "precision", true, edm::Comment("the precision with which to store the value in the flat table")) xor
            edm::ParameterDescription<std::string>(
                "precision", true, edm::Comment("the precision with which to store the value in the flat table")),
        false);
    edm::ParameterSetDescription colvariables;
    colvariables.setComment("a parameters set to define all variable to fill the flat table");
    colvariables.addNode(
        edm::ParameterWildcard<edm::ParameterSetDescription>("*", edm::RequireAtLeastOne, true, colvariable));

    edm::ParameterSetDescription coltable;
    coltable.addOptional<std::string>("name")->setComment(
        "name of the branch in the flat table containing flatten collections of variables");
    coltable.add<std::string>("doc")->setComment(
        "few words description of the table containing flatten collections of variables");
    coltable.add<bool>("useCount", true)
        ->setComment("whether to use count for the main table to index table with flatten collections of variables");
    coltable.add<bool>("useOffset", false)
        ->setComment("whether to use offset for the main table to index table with flatten collections of variables");
    coltable.add<edm::ParameterSetDescription>("variables", colvariables);

    edm::ParameterSetDescription coltables;
    coltables.setComment("a parameters set to define variables to be flatten to fill the table");
    coltables.addOptionalNode(
        edm::ParameterWildcard<edm::ParameterSetDescription>("*", edm::RequireZeroOrMore, true, coltable), false);
    desc.addOptional<edm::ParameterSetDescription>("collectionVariables", coltables);

    descriptions.addWithDefaultLabel(desc);
  }

protected:
  template <typename R>
  using VectorVar = DummyCollectionVariable<T, R>;

  using IntVectorVar = VectorVar<int32_t>;
  using UIntVectorVar = VectorVar<uint32_t>;
  using FloatVectorVar = VectorVar<float>;
  using DoubleVectorVar = VectorVar<double>;
  using UInt8VectorVar = VectorVar<uint8_t>;
  using Int16VectorVar = VectorVar<int16_t>;
  using UInt16VectorVar = VectorVar<uint16_t>;

  struct CollectionVariableTableInfo {
    std::string name;
    std::string doc;
    bool useCount;
    bool useOffset;
    std::vector<std::unique_ptr<CollectionVariable<T>>> colvars;
  };
  std::vector<CollectionVariableTableInfo> coltables;
};

typedef ticl::AssociationMap<vector<vector<ticl::AssociationElement<pair<ticl::SharedEnergyType, float>>>>,
                             vector<ticl::Trackster>,
                             vector<ticl::Trackster>>
    associationMap;
typedef SimpleAssociationCollectionFlatTableProducer<associationMap> TracksterAssociationCollectionTableProducer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TracksterAssociationCollectionTableProducer);
