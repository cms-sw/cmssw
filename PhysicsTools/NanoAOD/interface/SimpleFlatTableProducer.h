#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"

#include <memory>
#include <vector>

template <typename T, typename TProd>
class SimpleFlatTableProducerBase : public edm::stream::EDProducer<> {
public:
  SimpleFlatTableProducerBase(edm::ParameterSet const &params)
      : name_(params.getParameter<std::string>("name")),
        doc_(params.existsAs<std::string>("doc") ? params.getParameter<std::string>("doc") : ""),
        extension_(params.existsAs<bool>("extension") ? params.getParameter<bool>("extension") : false),
        skipNonExistingSrc_(
            params.existsAs<bool>("skipNonExistingSrc") ? params.getParameter<bool>("skipNonExistingSrc") : false),
        src_(skipNonExistingSrc_ ? mayConsume<TProd>(params.getParameter<edm::InputTag>("src"))
                                 : consumes<TProd>(params.getParameter<edm::InputTag>("src"))) {
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

    produces<nanoaod::FlatTable>();
  }

  ~SimpleFlatTableProducerBase() override {}

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

  class Variable : public VariableBase {
  public:
    Variable(const std::string &aname, const edm::ParameterSet &cfg) : VariableBase(aname, cfg) {}
    virtual void fill(std::vector<const T *> selobjs, nanoaod::FlatTable &out) const = 0;
  };

  template <typename StringFunctor, typename ValType>
  class FuncVariable : public Variable {
  public:
    FuncVariable(const std::string &aname, const edm::ParameterSet &cfg)
        : Variable(aname, cfg),
          func_(cfg.getParameter<std::string>("expr"), true),
          precisionFunc_(cfg.existsAs<std::string>("precision") ? cfg.getParameter<std::string>("precision") : "23",
                         true) {}
    ~FuncVariable() override {}
    void fill(std::vector<const T *> selobjs, nanoaod::FlatTable &out) const override {
      std::vector<ValType> vals(selobjs.size());
      for (unsigned int i = 0, n = vals.size(); i < n; ++i) {
        if constexpr (std::is_same<ValType, float>()) {
          if (this->precision_ == -2) {
            vals[i] =
                MiniFloatConverter::reduceMantissaToNbitsRounding(func_(*selobjs[i]), precisionFunc_(*selobjs[i]));
          } else {
            vals[i] = func_(*selobjs[i]);
          }
        } else {
          vals[i] = func_(*selobjs[i]);
        }
      }
      out.template addColumn<ValType>(this->name_, vals, this->doc_, this->precision_);
    }

  protected:
    StringFunctor func_;
    StringFunctor precisionFunc_;
  };
  typedef FuncVariable<StringObjectFunction<T>, int> IntVar;
  typedef FuncVariable<StringObjectFunction<T>, float> FloatVar;
  typedef FuncVariable<StringObjectFunction<T>, uint8_t> UInt8Var;
  typedef FuncVariable<StringCutObjectSelector<T>, bool> BoolVar;
  std::vector<std::unique_ptr<Variable>> vars_;
};

template <typename T>
class SimpleFlatTableProducer : public SimpleFlatTableProducerBase<T, edm::View<T>> {
public:
  typedef SimpleFlatTableProducerBase<T, edm::View<T>> base;

  SimpleFlatTableProducer(edm::ParameterSet const &params)
      : SimpleFlatTableProducerBase<T, edm::View<T>>(params),
        singleton_(params.getParameter<bool>("singleton")),
        maxLen_(params.existsAs<unsigned int>("maxLen") ? params.getParameter<unsigned int>("maxLen")
                                                        : std::numeric_limits<unsigned int>::max()),
        cut_(!singleton_ ? params.getParameter<std::string>("cut") : "", true) {
    if (params.existsAs<edm::ParameterSet>("externalVariables")) {
      edm::ParameterSet const &extvarsPSet = params.getParameter<edm::ParameterSet>("externalVariables");
      for (const std::string &vname : extvarsPSet.getParameterNamesForType<edm::ParameterSet>()) {
        const auto &varPSet = extvarsPSet.getParameter<edm::ParameterSet>(vname);
        const std::string &type = varPSet.getParameter<std::string>("type");
        if (type == "int")
          extvars_.push_back(
              std::make_unique<IntExtVar>(vname, varPSet, this->consumesCollector(), this->skipNonExistingSrc_));
        else if (type == "float")
          extvars_.push_back(
              std::make_unique<FloatExtVar>(vname, varPSet, this->consumesCollector(), this->skipNonExistingSrc_));
        else if (type == "double")
          extvars_.push_back(
              std::make_unique<DoubleExtVar>(vname, varPSet, this->consumesCollector(), this->skipNonExistingSrc_));
        else if (type == "uint8")
          extvars_.push_back(
              std::make_unique<UInt8ExtVar>(vname, varPSet, this->consumesCollector(), this->skipNonExistingSrc_));
        else if (type == "bool")
          extvars_.push_back(
              std::make_unique<BoolExtVar>(vname, varPSet, this->consumesCollector(), this->skipNonExistingSrc_));
        else
          throw cms::Exception("Configuration", "unsupported type " + type + " for variable " + vname);
      }
    }
  }

  ~SimpleFlatTableProducer() override {}

  std::unique_ptr<nanoaod::FlatTable> fillTable(const edm::Event &iEvent,
                                                const edm::Handle<edm::View<T>> &prod) const override {
    std::vector<const T *> selobjs;
    std::vector<edm::Ptr<T>> selptrs;  // for external variables
    if (prod.isValid() || !(this->skipNonExistingSrc_)) {
      if (singleton_) {
        assert(prod->size() == 1);
        selobjs.push_back(&(*prod)[0]);
        if (!extvars_.empty())
          selptrs.emplace_back(prod->ptrAt(0));
      } else {
        for (unsigned int i = 0, n = prod->size(); i < n; ++i) {
          const auto &obj = (*prod)[i];
          if (cut_(obj)) {
            selobjs.push_back(&obj);
            if (!extvars_.empty())
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
    return out;
  }

protected:
  bool singleton_;
  const unsigned int maxLen_;
  const StringCutObjectSelector<T> cut_;

  class ExtVariable : public base::VariableBase {
  public:
    ExtVariable(const std::string &aname, const edm::ParameterSet &cfg) : base::VariableBase(aname, cfg) {}
    virtual void fill(const edm::Event &iEvent, std::vector<edm::Ptr<T>> selptrs, nanoaod::FlatTable &out) const = 0;
  };
  template <typename TIn, typename ValType = TIn>
  class ValueMapVariable : public ExtVariable {
  public:
    ValueMapVariable(const std::string &aname,
                     const edm::ParameterSet &cfg,
                     edm::ConsumesCollector &&cc,
                     bool skipNonExistingSrc = false)
        : ExtVariable(aname, cfg),
          skipNonExistingSrc_(skipNonExistingSrc),
          token_(skipNonExistingSrc_ ? cc.mayConsume<edm::ValueMap<TIn>>(cfg.getParameter<edm::InputTag>("src"))
                                     : cc.consumes<edm::ValueMap<TIn>>(cfg.getParameter<edm::InputTag>("src"))) {}
    void fill(const edm::Event &iEvent, std::vector<edm::Ptr<T>> selptrs, nanoaod::FlatTable &out) const override {
      edm::Handle<edm::ValueMap<TIn>> vmap;
      iEvent.getByToken(token_, vmap);
      std::vector<ValType> vals;
      if (vmap.isValid() || !skipNonExistingSrc_) {
        vals.resize(selptrs.size());
        for (unsigned int i = 0, n = vals.size(); i < n; ++i) {
          vals[i] = (*vmap)[selptrs[i]];
        }
      }
      out.template addColumn<ValType>(this->name_, vals, this->doc_, this->precision_);
    }

  protected:
    const bool skipNonExistingSrc_;
    edm::EDGetTokenT<edm::ValueMap<TIn>> token_;
  };
  typedef ValueMapVariable<int> IntExtVar;
  typedef ValueMapVariable<float> FloatExtVar;
  typedef ValueMapVariable<double, float> DoubleExtVar;
  typedef ValueMapVariable<bool> BoolExtVar;
  typedef ValueMapVariable<int, uint8_t> UInt8ExtVar;
  std::vector<std::unique_ptr<ExtVariable>> extvars_;
};

template <typename T>
class EventSingletonSimpleFlatTableProducer : public SimpleFlatTableProducerBase<T, T> {
public:
  EventSingletonSimpleFlatTableProducer(edm::ParameterSet const &params) : SimpleFlatTableProducerBase<T, T>(params) {}

  ~EventSingletonSimpleFlatTableProducer() override {}

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

  std::unique_ptr<nanoaod::FlatTable> fillTable(const edm::Event &iEvent,
                                                const edm::Handle<edm::View<T>> &prod) const override {
    auto out = std::make_unique<nanoaod::FlatTable>(1, this->name_, true, this->extension_);
    std::vector<const T *> selobjs(1, &(*prod)[0]);
    for (const auto &var : this->vars_)
      var->fill(selobjs, *out);
    return out;
  }
};
