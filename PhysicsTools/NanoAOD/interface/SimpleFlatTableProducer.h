#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/EmptyGroupDescription.h"
#include "FWCore/ParameterSet/interface/allowedValues.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "Utilities/General/interface/ClassName.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"

#include <memory>
#include <vector>

// Base class for dumped variables
class VariableBase {
public:
  VariableBase(const std::string &aname, const edm::ParameterSet &cfg)
      : name_(aname),
        doc_(cfg.getParameter<std::string>("doc")),
        precision_(cfg.existsAs<int>("precision") ? cfg.getParameter<int>("precision")
                                                  : (cfg.existsAs<std::string>("precision") ? -2 : -1)) {}
  virtual ~VariableBase() {}
  const std::string &name() const { return name_; }

protected:
  std::string name_, doc_;
  int precision_;
};

// Object member variables and methods
template <typename ObjType>
class Variable : public VariableBase {
public:
  Variable(const std::string &aname, const edm::ParameterSet &cfg) : VariableBase(aname, cfg) {}
  virtual void fill(std::vector<const ObjType *> &selobjs, nanoaod::FlatTable &out) const = 0;
};

template <typename ObjType, typename StringFunctor, typename ValType>
class FuncVariable : public Variable<ObjType> {
public:
  FuncVariable(const std::string &aname, const edm::ParameterSet &cfg)
      : Variable<ObjType>(aname, cfg),
        func_(cfg.getParameter<std::string>("expr"), cfg.getUntrackedParameter<bool>("lazyEval")),
        precisionFunc_(cfg.existsAs<std::string>("precision") ? cfg.getParameter<std::string>("precision") : "23",
                       cfg.getUntrackedParameter<bool>("lazyEval")) {}
  ~FuncVariable() override {}

  void fill(std::vector<const ObjType *> &selobjs, nanoaod::FlatTable &out) const override {
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
  }

protected:
  StringFunctor func_;
  StringFunctor precisionFunc_;
};

// External variables: i.e. variables that are not member or methods of the object
template <typename ObjType>
class ExtVariable : public VariableBase {
public:
  ExtVariable(const std::string &aname, const edm::ParameterSet &cfg) : VariableBase(aname, cfg) {}
  virtual void fill(const edm::Event &iEvent,
                    std::vector<edm::Ptr<ObjType>> selptrs,
                    nanoaod::FlatTable &out) const = 0;
};

template <typename ObjType, typename TIn, typename ValType = TIn>
class ValueMapVariableBase : public ExtVariable<ObjType> {
public:
  ValueMapVariableBase(const std::string &aname,
                       const edm::ParameterSet &cfg,
                       edm::ConsumesCollector &&cc,
                       bool skipNonExistingSrc = false)
      : ExtVariable<ObjType>(aname, cfg),
        skipNonExistingSrc_(skipNonExistingSrc),
        token_(cc.consumes<edm::ValueMap<TIn>>(cfg.getParameter<edm::InputTag>("src"))) {}
  virtual ValType eval(const edm::Handle<edm::ValueMap<TIn>> &vmap, const edm::Ptr<ObjType> &op) const = 0;
  void fill(const edm::Event &iEvent, std::vector<edm::Ptr<ObjType>> selptrs, nanoaod::FlatTable &out) const override {
    edm::Handle<edm::ValueMap<TIn>> vmap;
    iEvent.getByToken(token_, vmap);
    std::vector<ValType> vals;
    if (vmap.isValid() || !skipNonExistingSrc_) {
      vals.resize(selptrs.size());
      for (unsigned int i = 0, n = vals.size(); i < n; ++i) {
        // calls the overloaded method to either get the valuemap value directly, or a function of the object value.
        vals[i] = this->eval(vmap, selptrs[i]);
      }
    }
    out.template addColumn<ValType>(this->name_, vals, this->doc_, this->precision_);
  }

protected:
  const bool skipNonExistingSrc_;
  edm::EDGetTokenT<edm::ValueMap<TIn>> token_;
};

template <typename ObjType, typename TIn, typename ValType = TIn>
class ValueMapVariable : public ValueMapVariableBase<ObjType, TIn, ValType> {
public:
  ValueMapVariable(const std::string &aname,
                   const edm::ParameterSet &cfg,
                   edm::ConsumesCollector &&cc,
                   bool skipNonExistingSrc = false)
      : ValueMapVariableBase<ObjType, TIn, ValType>(aname, cfg, std::move(cc), skipNonExistingSrc) {}
  ValType eval(const edm::Handle<edm::ValueMap<TIn>> &vmap, const edm::Ptr<ObjType> &op) const override {
    ValType val = (*vmap)[op];
    return val;
  }
};

template <typename ObjType, typename TIn, typename StringFunctor, typename ValType>
class TypedValueMapVariable : public ValueMapVariableBase<ObjType, TIn, ValType> {
public:
  TypedValueMapVariable(const std::string &aname,
                        const edm::ParameterSet &cfg,
                        edm::ConsumesCollector &&cc,
                        bool skipNonExistingSrc = false)
      : ValueMapVariableBase<ObjType, TIn, ValType>(aname, cfg, std::move(cc), skipNonExistingSrc),
        func_(cfg.getParameter<std::string>("expr"), true),
        precisionFunc_(cfg.existsAs<std::string>("precision") ? cfg.getParameter<std::string>("precision") : "23",
                       true) {}

  ValType eval(const edm::Handle<edm::ValueMap<TIn>> &vmap, const edm::Ptr<ObjType> &op) const override {
    ValType val = func_((*vmap)[op]);
    if constexpr (std::is_same<ValType, float>()) {
      if (this->precision_ == -2) {
        auto prec = precisionFunc_(*op);
        if (prec > 0) {
          val = MiniFloatConverter::reduceMantissaToNbitsRounding(val, prec);
        }
      }
    }
    return val;
  }

protected:
  StringFunctor func_;
  StringObjectFunction<ObjType> precisionFunc_;
};

// Event producers
// - ABC
// - Singleton
// - Collection
template <typename T, typename TProd>
class SimpleFlatTableProducerBase : public edm::stream::EDProducer<> {
public:
  SimpleFlatTableProducerBase(edm::ParameterSet const &params)
      : name_(params.getParameter<std::string>("name")),
        doc_(params.getParameter<std::string>("doc")),
        extension_(params.getParameter<bool>("extension")),
        skipNonExistingSrc_(params.getParameter<bool>("skipNonExistingSrc")),
        src_(consumes<TProd>(params.getParameter<edm::InputTag>("src"))) {
    edm::ParameterSet const &varsPSet = params.getParameter<edm::ParameterSet>("variables");
    for (const std::string &vname : varsPSet.getParameterNamesForType<edm::ParameterSet>()) {
      const auto &varPSet = varsPSet.getParameter<edm::ParameterSet>(vname);
      const std::string &type = varPSet.getParameter<std::string>("type");
      if (type == "int")
        vars_.push_back(std::make_unique<IntVar>(vname, varPSet));
      else if (type == "uint")
        vars_.push_back(std::make_unique<UIntVar>(vname, varPSet));
      else if (type == "float")
        vars_.push_back(std::make_unique<FloatVar>(vname, varPSet));
      else if (type == "double")
        vars_.push_back(std::make_unique<DoubleVar>(vname, varPSet));
      else if (type == "uint8")
        vars_.push_back(std::make_unique<UInt8Var>(vname, varPSet));
      else if (type == "int16")
        vars_.push_back(std::make_unique<Int16Var>(vname, varPSet));
      else if (type == "uint16")
        vars_.push_back(std::make_unique<UInt16Var>(vname, varPSet));
      else if (type == "bool")
        vars_.push_back(std::make_unique<BoolVar>(vname, varPSet));
      else
        throw cms::Exception("Configuration", "unsupported type " + type + " for variable " + vname);
    }

    produces<nanoaod::FlatTable>();
  }

  ~SimpleFlatTableProducerBase() override {}

  static edm::ParameterSetDescription baseDescriptions() {
    edm::ParameterSetDescription desc;
    std::string classname = ClassName<T>::name();
    desc.add<std::string>("name")->setComment("name of the branch in the flat table output for " + classname);
    desc.add<std::string>("doc", "")->setComment("few words of self documentation");
    desc.add<bool>("extension", false)->setComment("whether or not to extend an existing same table");
    desc.add<bool>("skipNonExistingSrc", false)
        ->setComment("whether or not to skip producing the table on absent input product");
    desc.add<edm::InputTag>("src")->setComment("input collection to fill the flat table");

    edm::ParameterSetDescription variable;
    variable.add<std::string>("expr")->setComment("a function to define the content of the branch in the flat table");
    variable.add<std::string>("doc")->setComment("few words description of the branch content");
    variable.addUntracked<bool>("lazyEval", false)
        ->setComment("if true, can use methods of inheriting classes in `expr`. Can cause problems with threading.");
    variable.ifValue(
        edm::ParameterDescription<std::string>(
            "type", "int", true, edm::Comment("the c++ type of the branch in the flat table")),
        edm::allowedValues<std::string>("int", "uint", "float", "double", "uint8", "int16", "uint16", "bool"));
    variable.addOptionalNode(
        edm::ParameterDescription<int>(
            "precision", true, edm::Comment("the precision with which to store the value in the flat table")) xor
            edm::ParameterDescription<std::string>(
                "precision", true, edm::Comment("the precision with which to store the value in the flat table")),
        false);

    edm::ParameterSetDescription variables;
    variables.setComment("a parameters set to define all variable to fill the flat table");
    variables.addNode(
        edm::ParameterWildcard<edm::ParameterSetDescription>("*", edm::RequireZeroOrMore, true, variable));
    desc.add<edm::ParameterSetDescription>("variables", variables);

    return desc;
  }
  // this is to be overriden by the child class
  virtual std::unique_ptr<nanoaod::FlatTable> fillTable(const edm::Event &iEvent,
                                                        const edm::Handle<TProd> &prod) const = 0;

  void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) override {
    edm::Handle<TProd> src;
    iEvent.getByToken(src_, src);

    std::unique_ptr<nanoaod::FlatTable> out = fillTable(iEvent, src);
    out->setDoc(doc_);

    iEvent.put(std::move(out));
  }

protected:
  const std::string name_;
  const std::string doc_;
  const bool extension_;
  const bool skipNonExistingSrc_;
  const edm::EDGetTokenT<TProd> src_;

  typedef FuncVariable<T, StringObjectFunction<T>, int32_t> IntVar;
  typedef FuncVariable<T, StringObjectFunction<T>, uint32_t> UIntVar;
  typedef FuncVariable<T, StringObjectFunction<T>, float> FloatVar;
  typedef FuncVariable<T, StringObjectFunction<T>, double> DoubleVar;
  typedef FuncVariable<T, StringObjectFunction<T>, uint8_t> UInt8Var;
  typedef FuncVariable<T, StringObjectFunction<T>, int16_t> Int16Var;
  typedef FuncVariable<T, StringObjectFunction<T>, uint16_t> UInt16Var;
  typedef FuncVariable<T, StringCutObjectSelector<T>, bool> BoolVar;
  std::vector<std::unique_ptr<Variable<T>>> vars_;
};

template <typename T>
class SimpleFlatTableProducer : public SimpleFlatTableProducerBase<T, edm::View<T>> {
public:
  SimpleFlatTableProducer(edm::ParameterSet const &params)
      : SimpleFlatTableProducerBase<T, edm::View<T>>(params),
        singleton_(params.getParameter<bool>("singleton")),
        maxLen_(params.existsAs<unsigned int>("maxLen") ? params.getParameter<unsigned int>("maxLen")
                                                        : std::numeric_limits<unsigned int>::max()),
        cut_(!singleton_ ? params.getParameter<std::string>("cut") : "",
             !singleton_ ? params.getUntrackedParameter<bool>("lazyEval") : false) {
    if (params.existsAs<edm::ParameterSet>("externalVariables")) {
      edm::ParameterSet const &extvarsPSet = params.getParameter<edm::ParameterSet>("externalVariables");
      for (const std::string &vname : extvarsPSet.getParameterNamesForType<edm::ParameterSet>()) {
        const auto &varPSet = extvarsPSet.getParameter<edm::ParameterSet>(vname);
        const std::string &type = varPSet.getParameter<std::string>("type");
        if (type == "int")
          extvars_.push_back(
              std::make_unique<IntExtVar>(vname, varPSet, this->consumesCollector(), this->skipNonExistingSrc_));
        else if (type == "uint")
          extvars_.push_back(
              std::make_unique<UIntExtVar>(vname, varPSet, this->consumesCollector(), this->skipNonExistingSrc_));
        else if (type == "float")
          extvars_.push_back(
              std::make_unique<FloatExtVar>(vname, varPSet, this->consumesCollector(), this->skipNonExistingSrc_));
        else if (type == "double")
          extvars_.push_back(
              std::make_unique<DoubleExtVar>(vname, varPSet, this->consumesCollector(), this->skipNonExistingSrc_));
        else if (type == "uint8")
          extvars_.push_back(
              std::make_unique<UInt8ExtVar>(vname, varPSet, this->consumesCollector(), this->skipNonExistingSrc_));
        else if (type == "int16")
          extvars_.push_back(
              std::make_unique<Int16ExtVar>(vname, varPSet, this->consumesCollector(), this->skipNonExistingSrc_));
        else if (type == "uint16")
          extvars_.push_back(
              std::make_unique<UInt16ExtVar>(vname, varPSet, this->consumesCollector(), this->skipNonExistingSrc_));
        else if (type == "bool")
          extvars_.push_back(
              std::make_unique<BoolExtVar>(vname, varPSet, this->consumesCollector(), this->skipNonExistingSrc_));
        else
          throw cms::Exception("Configuration", "unsupported type " + type + " for variable " + vname);
      }
    }
  }

  ~SimpleFlatTableProducer() override {}

  static edm::ParameterSetDescription baseDescriptions() {
    edm::ParameterSetDescription desc = SimpleFlatTableProducerBase<T, edm::View<T>>::baseDescriptions();

    desc.ifValue(
        edm::ParameterDescription<bool>(
            "singleton", false, true, edm::Comment("whether or not the input collection is single-element")),
        false >> (edm::ParameterDescription<std::string>(
                      "cut", "", true, edm::Comment("selection on the main input collection")) and
                  edm::ParameterDescription<bool>("lazyEval",
                                                  false,
                                                  false,
                                                  edm::Comment("if true, can use methods of inheriting classes. Can "
                                                               "cause problems when multi-threading."))) or
            true >> edm::EmptyGroupDescription());
    desc.addOptional<unsigned int>("maxLen")->setComment(
        "define the maximum length of the input collection to put in the branch");

    edm::ParameterSetDescription extvariable;
    extvariable.add<edm::InputTag>("src")->setComment("valuemap input collection to fill the flat table");
    extvariable.add<std::string>("doc")->setComment("few words description of the branch content");
    extvariable.ifValue(
        edm::ParameterDescription<std::string>(
            "type", "int", true, edm::Comment("the c++ type of the branch in the flat table")),
        edm::allowedValues<std::string>("int", "uint", "float", "double", "uint8", "int16", "uint16", "bool"));
    extvariable.addOptionalNode(
        edm::ParameterDescription<int>(
            "precision", true, edm::Comment("the precision with which to store the value in the flat table")) xor
            edm::ParameterDescription<std::string>("precision",
                                                   true,
                                                   edm::Comment("the precision with which to store the value in the "
                                                                "flat table, as a function of the object evaluated")),
        false);

    edm::ParameterSetDescription extvariables;
    extvariables.setComment("a parameters set to define all variable taken form valuemap to fill the flat table");
    extvariables.addOptionalNode(
        edm::ParameterWildcard<edm::ParameterSetDescription>("*", edm::RequireZeroOrMore, true, extvariable), false);
    desc.addOptional<edm::ParameterSetDescription>("externalVariables", extvariables);

    return desc;
  }
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc = SimpleFlatTableProducer<T>::baseDescriptions();
    descriptions.addWithDefaultLabel(desc);
  }
  std::unique_ptr<nanoaod::FlatTable> fillTable(const edm::Event &iEvent,
                                                const edm::Handle<edm::View<T>> &prod) const override {
    std::vector<const T *> selobjs;
    std::vector<edm::Ptr<T>> selptrs;  // for external variables
    if (prod.isValid() || !(this->skipNonExistingSrc_)) {
      if (singleton_) {
        assert(prod->size() == 1);
        selobjs.push_back(&(*prod)[0]);
        if (!extvars_.empty() || !typedextvars_.empty())
          selptrs.emplace_back(prod->ptrAt(0));
      } else {
        for (unsigned int i = 0, n = prod->size(); i < n; ++i) {
          const auto &obj = (*prod)[i];
          if (cut_(obj)) {
            selobjs.push_back(&obj);
            if (!extvars_.empty() || !typedextvars_.empty())
              selptrs.emplace_back(prod->ptrAt(i));
          }
          if (selobjs.size() >= maxLen_)
            break;
        }
      }
    }
    auto out = std::make_unique<nanoaod::FlatTable>(selobjs.size(), this->name_, singleton_, this->extension_);
    for (const auto &var : this->vars_)
      var->fill(selobjs, *out);
    for (const auto &var : this->extvars_)
      var->fill(iEvent, selptrs, *out);
    for (const auto &var : this->typedextvars_)
      var->fill(iEvent, selptrs, *out);
    return out;
  }

protected:
  bool singleton_;
  const unsigned int maxLen_;
  const StringCutObjectSelector<T> cut_;

  typedef ValueMapVariable<T, int32_t> IntExtVar;
  typedef ValueMapVariable<T, uint32_t> UIntExtVar;
  typedef ValueMapVariable<T, float> FloatExtVar;
  typedef ValueMapVariable<T, double, float> DoubleExtVar;
  typedef ValueMapVariable<T, bool> BoolExtVar;
  typedef ValueMapVariable<T, int, uint8_t> UInt8ExtVar;
  typedef ValueMapVariable<T, int, int16_t> Int16ExtVar;
  typedef ValueMapVariable<T, int, uint16_t> UInt16ExtVar;
  std::vector<std::unique_ptr<ExtVariable<T>>> extvars_;
  std::vector<std::unique_ptr<ExtVariable<T>>> typedextvars_;
};

template <typename T, typename V>
class SimpleTypedExternalFlatTableProducer : public SimpleFlatTableProducer<T> {
public:
  SimpleTypedExternalFlatTableProducer(edm::ParameterSet const &params) : SimpleFlatTableProducer<T>(params) {
    edm::ParameterSet const &extvarsPSet = params.getParameter<edm::ParameterSet>("externalTypedVariables");
    for (const std::string &vname : extvarsPSet.getParameterNamesForType<edm::ParameterSet>()) {
      const auto &varPSet = extvarsPSet.getParameter<edm::ParameterSet>(vname);
      const std::string &type = varPSet.getParameter<std::string>("type");
      if (type == "int")
        this->typedextvars_.push_back(
            std::make_unique<IntTypedExtVar>(vname, varPSet, this->consumesCollector(), this->skipNonExistingSrc_));
      else if (type == "uint")
        this->typedextvars_.push_back(
            std::make_unique<UIntTypedExtVar>(vname, varPSet, this->consumesCollector(), this->skipNonExistingSrc_));
      else if (type == "float")
        this->typedextvars_.push_back(
            std::make_unique<FloatTypedExtVar>(vname, varPSet, this->consumesCollector(), this->skipNonExistingSrc_));
      else if (type == "double")
        this->typedextvars_.push_back(
            std::make_unique<DoubleTypedExtVar>(vname, varPSet, this->consumesCollector(), this->skipNonExistingSrc_));
      else if (type == "uint8")
        this->typedextvars_.push_back(
            std::make_unique<UInt8TypedExtVar>(vname, varPSet, this->consumesCollector(), this->skipNonExistingSrc_));
      else if (type == "int16")
        this->typedextvars_.push_back(
            std::make_unique<Int16TypedExtVar>(vname, varPSet, this->consumesCollector(), this->skipNonExistingSrc_));
      else if (type == "uint16")
        this->typedextvars_.push_back(
            std::make_unique<UInt16TypedExtVar>(vname, varPSet, this->consumesCollector(), this->skipNonExistingSrc_));
      else if (type == "bool")
        this->typedextvars_.push_back(
            std::make_unique<BoolTypedExtVar>(vname, varPSet, this->consumesCollector(), this->skipNonExistingSrc_));
      else
        throw cms::Exception("Configuration", "unsupported type " + type + " for variable " + vname);
    }
  }
  ~SimpleTypedExternalFlatTableProducer() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc = SimpleFlatTableProducer<T>::baseDescriptions();
    edm::ParameterSetDescription extvariable;
    extvariable.add<edm::InputTag>("src")->setComment("valuemap input collection to fill the flat table");
    extvariable.add<std::string>("expr")->setComment(
        "a function to define the content of the branch in the flat table");
    extvariable.add<std::string>("doc")->setComment("few words description of the branch content");
    extvariable.addUntracked<bool>("lazyEval", false)
        ->setComment("if true, can use methods of inheriting classes in `expr`. Can cause problems with threading.");
    extvariable.ifValue(
        edm::ParameterDescription<std::string>(
            "type", "int", true, edm::Comment("the c++ type of the branch in the flat table")),
        edm::allowedValues<std::string>("int", "uint", "float", "double", "uint8", "int16", "uint16", "bool"));
    extvariable.addOptionalNode(
        edm::ParameterDescription<int>(
            "precision", true, edm::Comment("the precision with which to store the value in the flat table")) xor
            edm::ParameterDescription<std::string>("precision",
                                                   true,
                                                   edm::Comment("the precision with which to store the value in the "
                                                                "flat table, as a function of the object evaluated")),
        false);

    edm::ParameterSetDescription extvariables;
    extvariables.setComment("a parameters set to define all variable taken form valuemap to fill the flat table");
    extvariables.addOptionalNode(
        edm::ParameterWildcard<edm::ParameterSetDescription>("*", edm::RequireZeroOrMore, true, extvariable), false);
    desc.addOptional<edm::ParameterSetDescription>("externalTypedVariables", extvariables);

    descriptions.addWithDefaultLabel(desc);
  }

protected:
  typedef TypedValueMapVariable<T, V, StringObjectFunction<V>, int32_t> IntTypedExtVar;
  typedef TypedValueMapVariable<T, V, StringObjectFunction<V>, uint32_t> UIntTypedExtVar;
  typedef TypedValueMapVariable<T, V, StringObjectFunction<V>, float> FloatTypedExtVar;
  typedef TypedValueMapVariable<T, V, StringObjectFunction<V>, double> DoubleTypedExtVar;
  typedef TypedValueMapVariable<T, V, StringCutObjectSelector<V>, bool> BoolTypedExtVar;
  typedef TypedValueMapVariable<T, V, StringObjectFunction<V>, uint8_t> UInt8TypedExtVar;
  typedef TypedValueMapVariable<T, V, StringObjectFunction<V>, int16_t> Int16TypedExtVar;
  typedef TypedValueMapVariable<T, V, StringObjectFunction<V>, uint16_t> UInt16TypedExtVar;
};

template <typename T>
class BXVectorSimpleFlatTableProducer : public SimpleFlatTableProducerBase<T, BXVector<T>> {
public:
  BXVectorSimpleFlatTableProducer(edm::ParameterSet const &params)
      : SimpleFlatTableProducerBase<T, BXVector<T>>(params),
        maxLen_(params.existsAs<unsigned int>("maxLen") ? params.getParameter<unsigned int>("maxLen")
                                                        : std::numeric_limits<unsigned int>::max()),
        cut_(params.getParameter<std::string>("cut"), false),
        minBX_(params.getParameter<int>("minBX")),
        maxBX_(params.getParameter<int>("maxBX")),
        alwaysWriteBXValue_(params.getParameter<bool>("alwaysWriteBXValue")),
        bxVarName_("bx") {
    edm::ParameterSet const &varsPSet = params.getParameter<edm::ParameterSet>("variables");
    auto varNames = varsPSet.getParameterNamesForType<edm::ParameterSet>();
    if (std::find(varNames.begin(), varNames.end(), bxVarName_) != varNames.end()) {
      throw cms::Exception("Configuration",
                           "BXVectorSimpleFlatTableProducer already defines the " + bxVarName_ +
                               "internally and thus you should not specify it yourself");
    }
  }

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc = SimpleFlatTableProducerBase<T, BXVector<T>>::baseDescriptions();
    desc.add<std::string>("cut", "")->setComment(
        "selection on the main input collection (but selection can not be bx based)");
    desc.addOptional<unsigned int>("maxLen")->setComment(
        "define the maximum length of the input collection to put in the branch");
    desc.add<int>("minBX", -2)->setComment("min bx (inclusive) to include");
    desc.add<int>("maxBX", 2)->setComment("max bx (inclusive) to include");
    desc.add<bool>("alwaysWriteBXValue", true)
        ->setComment("always write the bx number (event  when only one bx can be present, ie minBX==maxBX)");
    descriptions.addWithDefaultLabel(desc);
  }

  std::unique_ptr<nanoaod::FlatTable> fillTable(const edm::Event &iEvent,
                                                const edm::Handle<BXVector<T>> &prod) const override {
    std::vector<const T *> selObjs;
    std::vector<int> selObjBXs;

    if (prod.isValid() || !(this->skipNonExistingSrc_)) {
      const int minBX = std::max(minBX_, prod->getFirstBX());
      const int maxBX = std::min(maxBX_, prod->getLastBX());
      for (int bx = minBX; bx <= maxBX; bx++) {
        for (size_t objNr = 0, nrObjs = prod->size(bx); objNr < nrObjs; ++objNr) {
          const auto &obj = prod->at(bx, objNr);
          if (cut_(obj)) {
            selObjs.push_back(&obj);
            selObjBXs.push_back(bx);
          }
          if (selObjs.size() >= maxLen_)
            break;
        }
      }
    }
    auto out = std::make_unique<nanoaod::FlatTable>(selObjs.size(), this->name_, false, this->extension_);
    for (const auto &var : this->vars_)
      var->fill(selObjs, *out);
    if (alwaysWriteBXValue_ || minBX_ != maxBX_) {
      out->template addColumn<int16_t>(bxVarName_, selObjBXs, "BX of the L1 candidate");
    }
    return out;
  }

protected:
  const unsigned int maxLen_;
  const StringCutObjectSelector<T> cut_;
  const int minBX_;
  const int maxBX_;
  const bool alwaysWriteBXValue_;
  const std::string bxVarName_;
};

template <typename T>
class EventSingletonSimpleFlatTableProducer : public SimpleFlatTableProducerBase<T, T> {
public:
  EventSingletonSimpleFlatTableProducer(edm::ParameterSet const &params) : SimpleFlatTableProducerBase<T, T>(params) {}

  ~EventSingletonSimpleFlatTableProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc = SimpleFlatTableProducerBase<T, T>::baseDescriptions();
    descriptions.addWithDefaultLabel(desc);
  }

  std::unique_ptr<nanoaod::FlatTable> fillTable(const edm::Event &, const edm::Handle<T> &prod) const override {
    auto out = std::make_unique<nanoaod::FlatTable>(1, this->name_, true, this->extension_);
    std::vector<const T *> selobjs(1, prod.product());
    for (const auto &var : this->vars_)
      var->fill(selobjs, *out);
    return out;
  }
};

template <typename T>
class FirstObjectSimpleFlatTableProducer : public SimpleFlatTableProducerBase<T, edm::View<T>> {
public:
  FirstObjectSimpleFlatTableProducer(edm::ParameterSet const &params)
      : SimpleFlatTableProducerBase<T, edm::View<T>>(params) {}

  ~FirstObjectSimpleFlatTableProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc = SimpleFlatTableProducerBase<T, edm::View<T>>::baseDescriptions();
    descriptions.addWithDefaultLabel(desc);
  }

  std::unique_ptr<nanoaod::FlatTable> fillTable(const edm::Event &iEvent,
                                                const edm::Handle<edm::View<T>> &prod) const override {
    auto out = std::make_unique<nanoaod::FlatTable>(1, this->name_, true, this->extension_);
    std::vector<const T *> selobjs(1, &(*prod)[0]);
    for (const auto &var : this->vars_)
      var->fill(selobjs, *out);
    return out;
  }
};

// LuminosityBlock producers
// - ABC
// - Singleton
// - Collection
template <typename T, typename TProd>
class SimpleFlatTableProducerBaseLumi
    : public edm::one::EDProducer<edm::EndLuminosityBlockProducer, edm::LuminosityBlockCache<int>> {
public:
  SimpleFlatTableProducerBaseLumi(edm::ParameterSet const &params)
      : name_(params.getParameter<std::string>("name")),
        doc_(params.existsAs<std::string>("doc") ? params.getParameter<std::string>("doc") : ""),
        extension_(params.existsAs<bool>("extension") ? params.getParameter<bool>("extension") : false),
        skipNonExistingSrc_(

            params.existsAs<bool>("skipNonExistingSrc") ? params.getParameter<bool>("skipNonExistingSrc") : false),
        src_(consumes<TProd, edm::InLumi>(params.getParameter<edm::InputTag>("src"))) {
    edm::ParameterSet const &varsPSet = params.getParameter<edm::ParameterSet>("variables");
    for (const std::string &vname : varsPSet.getParameterNamesForType<edm::ParameterSet>()) {
      const auto &varPSet = varsPSet.getParameter<edm::ParameterSet>(vname);
      const std::string &type = varPSet.getParameter<std::string>("type");
      if (type == "int")
        vars_.push_back(std::make_unique<IntVar>(vname, varPSet));
      else if (type == "float")
        vars_.push_back(std::make_unique<FloatVar>(vname, varPSet));
      else if (type == "uint8")
        vars_.push_back(std::make_unique<UInt8Var>(vname, varPSet));
      else if (type == "bool")
        vars_.push_back(std::make_unique<BoolVar>(vname, varPSet));
      else
        throw cms::Exception("Configuration", "unsupported type " + type + " for variable " + vname);
    }

    produces<nanoaod::FlatTable, edm::Transition::EndLuminosityBlock>();
  }

  ~SimpleFlatTableProducerBaseLumi() override {}

  std::shared_ptr<int> globalBeginLuminosityBlock(edm::LuminosityBlock const &,
                                                  edm::EventSetup const &) const override {
    return nullptr;
  }

  void globalEndLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const &) override {}

  // this is to be overriden by the child class
  virtual std::unique_ptr<nanoaod::FlatTable> fillTable(const edm::LuminosityBlock &iLumi,
                                                        const edm::Handle<TProd> &prod) const = 0;

  void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) override {
    // do nothing
  }

  void endLuminosityBlockProduce(edm::LuminosityBlock &iLumi, const edm::EventSetup &iSetup) final {
    edm::Handle<TProd> src;
    iLumi.getByToken(src_, src);

    std::unique_ptr<nanoaod::FlatTable> out = fillTable(iLumi, src);
    out->setDoc(doc_);

    iLumi.put(std::move(out));
  }

protected:
  const std::string name_;
  const std::string doc_;
  const bool extension_;
  const bool skipNonExistingSrc_;
  const edm::EDGetTokenT<TProd> src_;

  typedef FuncVariable<T, StringObjectFunction<T>, int> IntVar;
  typedef FuncVariable<T, StringObjectFunction<T>, float> FloatVar;
  typedef FuncVariable<T, StringObjectFunction<T>, uint8_t> UInt8Var;
  typedef FuncVariable<T, StringCutObjectSelector<T>, bool> BoolVar;
  std::vector<std::unique_ptr<Variable<T>>> vars_;
};

// Class for singletons like GenFilterInfo
template <typename T>
class LumiSingletonSimpleFlatTableProducer : public SimpleFlatTableProducerBaseLumi<T, T> {
public:
  LumiSingletonSimpleFlatTableProducer(edm::ParameterSet const &params)
      : SimpleFlatTableProducerBaseLumi<T, T>(params) {}

  ~LumiSingletonSimpleFlatTableProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc = SimpleFlatTableProducerBase<T, T>::baseDescriptions();
    descriptions.addWithDefaultLabel(desc);
  }

  std::unique_ptr<nanoaod::FlatTable> fillTable(const edm::LuminosityBlock &,
                                                const edm::Handle<T> &prod) const override {
    auto out = std::make_unique<nanoaod::FlatTable>(1, this->name_, true, this->extension_);
    std::vector<const T *> selobjs(1, prod.product());
    for (const auto &var : this->vars_)
      var->fill(selobjs, *out);
    return out;
  }
};

// Class for generic collections
template <typename T, typename TProd>
class LumiSimpleFlatTableProducer : public SimpleFlatTableProducerBaseLumi<T, TProd> {
public:
  LumiSimpleFlatTableProducer(edm::ParameterSet const &params)
      : SimpleFlatTableProducerBaseLumi<T, TProd>(params),
        maxLen_(params.existsAs<unsigned int>("maxLen") ? params.getParameter<unsigned int>("maxLen")
                                                        : std::numeric_limits<unsigned int>::max()),
        cut_(params.existsAs<std::string>("cut") ? params.getParameter<std::string>("cut") : "", true) {}

  ~LumiSimpleFlatTableProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc = SimpleFlatTableProducerBase<T, TProd>::baseDescriptions();
    desc.addOptional<unsigned int>("maxLen")->setComment(
        "define the maximum length of the input collection to put in the branch");
    descriptions.addWithDefaultLabel(desc);
  }

  std::unique_ptr<nanoaod::FlatTable> fillTable(const edm::LuminosityBlock &iLumi,
                                                const edm::Handle<TProd> &prod) const override {
    std::vector<const T *> selobjs;
    if (prod.isValid() || !(this->skipNonExistingSrc_)) {
      for (unsigned int i = 0, n = prod->size(); i < n; ++i) {
        const auto &obj = (*prod)[i];
        if (cut_(obj)) {
          selobjs.push_back(&obj);
        }
        if (selobjs.size() >= maxLen_)
          break;
      }
    }
    auto out = std::make_unique<nanoaod::FlatTable>(selobjs.size(), this->name_, false, this->extension_);
    for (const auto &var : this->vars_)
      var->fill(selobjs, *out);
    return out;
  }

protected:
  const unsigned int maxLen_;
  const StringCutObjectSelector<T> cut_;
};

// Run producers
// - ABC
// - Singleton
// - Collection
template <typename T, typename TProd>
class SimpleFlatTableProducerBaseRun : public edm::one::EDProducer<edm::EndRunProducer, edm::RunCache<int>> {
public:
  SimpleFlatTableProducerBaseRun(edm::ParameterSet const &params)
      : name_(params.getParameter<std::string>("name")),
        doc_(params.existsAs<std::string>("doc") ? params.getParameter<std::string>("doc") : ""),
        extension_(params.existsAs<bool>("extension") ? params.getParameter<bool>("extension") : false),
        skipNonExistingSrc_(

            params.existsAs<bool>("skipNonExistingSrc") ? params.getParameter<bool>("skipNonExistingSrc") : false),
        src_(consumes<TProd, edm::InRun>(params.getParameter<edm::InputTag>("src"))) {
    edm::ParameterSet const &varsPSet = params.getParameter<edm::ParameterSet>("variables");
    for (const std::string &vname : varsPSet.getParameterNamesForType<edm::ParameterSet>()) {
      const auto &varPSet = varsPSet.getParameter<edm::ParameterSet>(vname);
      const std::string &type = varPSet.getParameter<std::string>("type");
      if (type == "int")
        vars_.push_back(std::make_unique<IntVar>(vname, varPSet));
      else if (type == "float")
        vars_.push_back(std::make_unique<FloatVar>(vname, varPSet));
      else if (type == "uint8")
        vars_.push_back(std::make_unique<UInt8Var>(vname, varPSet));
      else if (type == "bool")
        vars_.push_back(std::make_unique<BoolVar>(vname, varPSet));
      else
        throw cms::Exception("Configuration", "unsupported type " + type + " for variable " + vname);
    }

    produces<nanoaod::FlatTable, edm::Transition::EndRun>();
  }

  ~SimpleFlatTableProducerBaseRun() override {}

  std::shared_ptr<int> globalBeginRun(edm::Run const &, edm::EventSetup const &) const override { return nullptr; }

  void globalEndRun(edm::Run const &, edm::EventSetup const &) override {}

  // this is to be overriden by the child class
  virtual std::unique_ptr<nanoaod::FlatTable> fillTable(const edm::Run &iRun, const edm::Handle<TProd> &prod) const = 0;

  void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) override {
    // do nothing
  }

  void endRunProduce(edm::Run &iRun, const edm::EventSetup &iSetup) final {
    edm::Handle<TProd> src;
    iRun.getByToken(src_, src);

    std::unique_ptr<nanoaod::FlatTable> out = fillTable(iRun, src);
    out->setDoc(doc_);

    iRun.put(std::move(out));
  }

protected:
  const std::string name_;
  const std::string doc_;
  const bool extension_;
  const bool skipNonExistingSrc_;
  const edm::EDGetTokenT<TProd> src_;

  typedef FuncVariable<T, StringObjectFunction<T>, int> IntVar;
  typedef FuncVariable<T, StringObjectFunction<T>, float> FloatVar;
  typedef FuncVariable<T, StringObjectFunction<T>, uint8_t> UInt8Var;
  typedef FuncVariable<T, StringCutObjectSelector<T>, bool> BoolVar;
  std::vector<std::unique_ptr<Variable<T>>> vars_;
};

// Class for singletons like GenFilterInfo
template <typename T>
class RunSingletonSimpleFlatTableProducer : public SimpleFlatTableProducerBaseRun<T, T> {
public:
  RunSingletonSimpleFlatTableProducer(edm::ParameterSet const &params) : SimpleFlatTableProducerBaseRun<T, T>(params) {}

  ~RunSingletonSimpleFlatTableProducer() override {}

  std::unique_ptr<nanoaod::FlatTable> fillTable(const edm::Run &, const edm::Handle<T> &prod) const override {
    auto out = std::make_unique<nanoaod::FlatTable>(1, this->name_, true, this->extension_);
    std::vector<const T *> selobjs(1, prod.product());
    for (const auto &var : this->vars_)
      var->fill(selobjs, *out);
    return out;
  }
};

// Class for generic collections
template <typename T, typename TProd>
class RunSimpleFlatTableProducer : public SimpleFlatTableProducerBaseRun<T, TProd> {
public:
  RunSimpleFlatTableProducer(edm::ParameterSet const &params)
      : SimpleFlatTableProducerBaseRun<T, TProd>(params),
        maxLen_(params.existsAs<unsigned int>("maxLen") ? params.getParameter<unsigned int>("maxLen")
                                                        : std::numeric_limits<unsigned int>::max()),
        cut_(params.existsAs<std::string>("cut") ? params.getParameter<std::string>("cut") : "", true) {}

  ~RunSimpleFlatTableProducer() override {}

  std::unique_ptr<nanoaod::FlatTable> fillTable(const edm::Run &iRun, const edm::Handle<TProd> &prod) const override {
    std::vector<const T *> selobjs;
    if (prod.isValid() || !(this->skipNonExistingSrc_)) {
      for (unsigned int i = 0, n = prod->size(); i < n; ++i) {
        const auto &obj = (*prod)[i];
        if (cut_(obj)) {
          selobjs.push_back(&obj);
        }
        if (selobjs.size() >= maxLen_)
          break;
      }
    }
    auto out = std::make_unique<nanoaod::FlatTable>(selobjs.size(), this->name_, false, this->extension_);
    for (const auto &var : this->vars_)
      var->fill(selobjs, *out);
    return out;
  }

protected:
  const unsigned int maxLen_;
  const StringCutObjectSelector<T> cut_;
};
