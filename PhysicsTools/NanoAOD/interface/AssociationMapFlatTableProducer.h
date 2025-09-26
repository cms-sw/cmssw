#ifndef HLTrigger_HLTUpgradeNano_AssociationMapFlatTable_h
#define HLTrigger_HLTUpgradeNano_AssociationMapFlatTable_h

#include <type_traits>

#include "PhysicsTools/NanoAOD/interface/SimpleFlatTableProducer.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"

// Concept to check if a type is a valid AssociationMap of AssociationElement, both oneToOne and oneToMany.
// For oneToOne pass as Container the type std::vector<T::AssociationElementType>
// For oneToMany pass as Container the type std::vector<std::vector<T::AssociationElementType>>
template <typename T, typename Container>
concept IsValidAssociationMap = requires {
  typename T::Traits;
  typename T::AssociationElementType;
  typename T::V;
  typename T::Type;

  requires std::is_same_v<typename T::Type, Container>;
};

template <typename T>
  requires IsValidAssociationMap<T, std::vector<typename T::AssociationElementType>>
class AssociationOneToOneFlatTableProducer : public SimpleFlatTableProducerBase<typename T::AssociationElementType, T> {
public:
  using TProd = T::AssociationElementType;
  AssociationOneToOneFlatTableProducer(edm::ParameterSet const &params)
      : SimpleFlatTableProducerBase<typename T::AssociationElementType, T>(params) {}

  ~AssociationOneToOneFlatTableProducer() override {}

  std::unique_ptr<nanoaod::FlatTable> fillTable(const edm::Event &iEvent, const edm::Handle<T> &prod) const override {
    // First unroll the Container inside the associator map.

    auto table_size = prod->getMap().size();
    auto out = std::make_unique<nanoaod::FlatTable>(table_size, this->name_, false);

    std::vector<const TProd *> selobjs;
    if (prod.isValid() || !(this->skipNonExistingSrc_)) {
      for (unsigned int i = 0, n = prod->size(); i < n; ++i) {
        const auto &obj = (*prod)[i];
        selobjs.push_back(&obj);
      }
    }

    for (const auto &var : this->vars_)
      var->fill(selobjs, *out);
    return out;
  }

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc =
        SimpleFlatTableProducerBase<typename T::AssociationElementType, T>::baseDescriptions();
    descriptions.addWithDefaultLabel(desc);
  }
};

template <typename T>
  requires IsValidAssociationMap<T, std::vector<std::vector<typename T::AssociationElementType>>>
class AssociationOneToManyFlatTableProducer : public SimpleFlatTableProducerBase<T, T> {
public:
  using TProd = T::AssociationElementType;
  AssociationOneToManyFlatTableProducer(edm::ParameterSet const &params) : SimpleFlatTableProducerBase<T, T>(params) {
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

  ~AssociationOneToManyFlatTableProducer() override {}

  void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) override {
    // same as SimpleFlatTableProducer
    edm::Handle<T> prod;
    iEvent.getByToken(SimpleFlatTableProducerBase<T, T>::src_, prod);

    // First unroll the Container inside the associator map.

    auto table_size = prod->getMap().size();
    auto out = std::make_unique<nanoaod::FlatTable>(table_size, this->name_, false);

    // Now proceed with the variable-sized linked objects.
    unsigned int coltablesize = 0;
    std::vector<unsigned int> counts;
    counts.reserve(table_size);
    for (auto const &links : prod->getMap()) {
      counts.push_back(links.size());
      coltablesize += counts.back();
    }

    std::vector<const TProd *> selobjs;
    selobjs.reserve(coltablesize);
    if (prod.isValid() || !(this->skipNonExistingSrc_)) {
      for (unsigned int i = 0, n = prod->size(); i < n; ++i) {
        const auto &obj = (*prod)[i];
        for (auto const &assocElement : obj) {
          selobjs.push_back(&assocElement);
        }
      }
    }

    // collection variable tables
    for (const auto &coltable : this->coltables) {
      // add count branch if requested
      if (coltable.useCount)
        out->template addColumn<uint16_t>("n" + coltable.name, counts, "counts for " + coltable.name);
      // add offset branch if requested
      if (coltable.useOffset) {
        unsigned int offset = 0;
        std::vector<unsigned int> offsets;
        offsets.reserve(table_size);
        for (auto const &count : counts) {
          offsets.push_back(offset);
          offset += count;
        }
        out->template addColumn<uint16_t>("o" + coltable.name, offsets, "offsets for " + coltable.name);
      }

      std::unique_ptr<nanoaod::FlatTable> outcoltable =
          std::make_unique<nanoaod::FlatTable>(coltablesize, coltable.name, false, false);
      for (const auto &colvar : coltable.colvars) {
        colvar->fill(selobjs, *outcoltable);
      }
      outcoltable->setDoc(coltable.doc);
      iEvent.put(std::move(outcoltable), coltable.name + "Table");
    }

    // put the main table into the event
    out->setDoc(this->doc_);
    iEvent.put(std::move(out));
  }

  // Make compiler happy.
  std::unique_ptr<nanoaod::FlatTable> fillTable(const edm::Event &iEvent, const edm::Handle<T> &prod) const override {
    // Do nothing here and move all the heavy lifting in the produce method.
    auto out = std::make_unique<nanoaod::FlatTable>();
    return out;
  }

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc = SimpleFlatTableProducerBase<T, T>::baseDescriptions();
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
  using VectorVar = FuncVariable<TProd, StringObjectFunction<TProd>, R>;

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
    std::vector<std::unique_ptr<Variable<TProd>>> colvars;
  };
  std::vector<CollectionVariableTableInfo> coltables;
};

// OneToOne, Fraction and Fraction with Score
template <typename Source, typename Target>
using AssociationMapOneToOneFraction =
    ticl::AssociationMap<vector<ticl::AssociationElement<ticl::FractionType>>, std::vector<Source>, std::vector<Target>>;

template <typename Source, typename Target>
using AssociationMapOneToOneFractionScore =
    ticl::AssociationMap<vector<ticl::AssociationElement<std::pair<ticl::FractionType, float>>>,
                         std::vector<Source>,
                         std::vector<Target>>;

// OneToOne, SharedEnergy and SharedEnergy with Score
template <typename Source, typename Target>
using AssociationMapOneToOneSharedEnergy = ticl::
    AssociationMap<vector<ticl::AssociationElement<ticl::SharedEnergyType>>, std::vector<Source>, std::vector<Target>>;

template <typename Source, typename Target>
using AssociationMapOneToOneSharedEnergyScore =
    ticl::AssociationMap<vector<ticl::AssociationElement<std::pair<ticl::SharedEnergyType, float>>>,
                         std::vector<Source>,
                         std::vector<Target>>;

// OneToMany, Fraction and Fraction with Score
template <typename Source, typename Target>
using AssociationMapOneToManyFraction =
    ticl::AssociationMap<vector<vector<ticl::AssociationElement<ticl::FractionType>>>, vector<Source>, vector<Target>>;

template <typename Source, typename Target>
using AssociationMapOneToManyFractionScore =
    ticl::AssociationMap<vector<vector<ticl::AssociationElement<pair<ticl::FractionType, float>>>>,
                         vector<Source>,
                         vector<Target>>;

// OneToMany, SharedEnergy and SharedEnergy with Score
template <typename Source, typename Target>
using AssociationMapOneToManySharedEnergy = ticl::
    AssociationMap<vector<vector<ticl::AssociationElement<ticl::SharedEnergyType>>>, vector<Source>, vector<Target>>;

template <typename Source, typename Target>
using AssociationMapOneToManySharedEnergyScore =
    ticl::AssociationMap<vector<vector<ticl::AssociationElement<pair<ticl::SharedEnergyType, float>>>>,
                         vector<Source>,
                         vector<Target>>;

#endif
