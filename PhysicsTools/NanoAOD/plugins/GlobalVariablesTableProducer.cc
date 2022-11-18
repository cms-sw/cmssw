#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/allowedValues.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include <memory>
#include <utility>
#include <vector>

class GlobalVariablesTableProducer : public edm::stream::EDProducer<> {
public:
  GlobalVariablesTableProducer(edm::ParameterSet const& params)
      : name_(params.getParameter<std::string>("name")), extension_(params.getParameter<bool>("extension")) {
    edm::ParameterSet const& varsPSet = params.getParameter<edm::ParameterSet>("variables");
    for (const std::string& vname : varsPSet.getParameterNamesForType<edm::ParameterSet>()) {
      const auto& varPSet = varsPSet.getParameter<edm::ParameterSet>(vname);
      const std::string& type = varPSet.getParameter<std::string>("type");
      if (type == "int")
        vars_.push_back(std::make_unique<IntVar>(vname, varPSet, consumesCollector()));
      else if (type == "float")
        vars_.push_back(std::make_unique<FloatVar>(vname, varPSet, consumesCollector()));
      else if (type == "double")
        vars_.push_back(std::make_unique<DoubleVar>(vname, varPSet, consumesCollector()));
      else if (type == "bool")
        vars_.push_back(std::make_unique<BoolVar>(vname, varPSet, consumesCollector()));
      else if (type == "candidatescalarsum")
        vars_.push_back(std::make_unique<CandidateScalarSumVar>(vname, varPSet, consumesCollector()));
      else if (type == "candidatesize")
        vars_.push_back(std::make_unique<CandidateSizeVar>(vname, varPSet, consumesCollector()));
      else if (type == "candidatesummass")
        vars_.push_back(std::make_unique<CandidateSumMassVar>(vname, varPSet, consumesCollector()));
      else
        throw cms::Exception("Configuration", "unsupported type " + type + " for variable " + vname);
    }

    produces<nanoaod::FlatTable>();
  }

  ~GlobalVariablesTableProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("name", "")->setComment("name of the branch in the flat table output");
    desc.add<bool>("extension", false)->setComment("whether or not to extend an existing same table");
    edm::ParameterSetDescription variable;
    variable.ifValue(edm::ParameterDescription<std::string>(
                         "type", "int", true, edm::Comment("the c++ type of the branch in the flat table")),
                     edm::allowedValues<std::string>(
                         "int", "float", "double", "bool", "candidatescalarsum", "candidatesize", "candidatesummass"));
    variable.add<edm::InputTag>("src")->setComment("input collection for the branch");
    variable.add<std::string>("doc")->setComment("few words description of the branch content");
    variable.add<int>("precision", -1)->setComment("precision to store the information");
    edm::ParameterSetDescription variables;
    variables.setComment("a parameters set to define variable to fill the flat table");
    variables.addNode(
        edm::ParameterWildcard<edm::ParameterSetDescription>("*", edm::RequireZeroOrMore, true, variable));
    desc.add<edm::ParameterSetDescription>("variables", variables);

    descriptions.addWithDefaultLabel(desc);
  }
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override {
    auto out = std::make_unique<nanoaod::FlatTable>(1, this->name_, true, this->extension_);

    for (const auto& var : vars_)
      var->fill(iEvent, *out);

    iEvent.put(std::move(out));
  }

protected:
  class Variable {
  public:
    Variable(const std::string& aname, const edm::ParameterSet& cfg)
        : name_(aname), doc_(cfg.getParameter<std::string>("doc")), precision_(cfg.getParameter<int>("precision")) {}
    virtual void fill(const edm::Event& iEvent, nanoaod::FlatTable& out) const = 0;
    virtual ~Variable() {}
    const std::string& name() const { return name_; }

  protected:
    std::string name_, doc_;
    int precision_;
  };
  template <typename ValType>
  class Identity {
  public:
    static ValType convert(ValType x) { return x; }
  };
  template <typename ValType>
  class Size {
  public:
    static int convert(ValType x) { return x.size(); }
  };

  template <typename ColType, typename ValType>
  class Max {
  public:
    static ColType convert(ValType x) {
      ColType v = std::numeric_limits<ColType>::min();
      for (const auto& i : x)
        if (i > v)
          v = i;
      return v;
    }
  };
  template <typename ColType, typename ValType>
  class Min {
  public:
    static ColType convert(ValType x) {
      ColType v = std::numeric_limits<ColType>::max();
      for (const auto& i : x)
        if (i < v)
          v = i;
      return v;
    }
  };
  template <typename ColType, typename ValType>
  class ScalarPtSum {
  public:
    static ColType convert(ValType x) {
      ColType v = 0;
      for (const auto& i : x)
        v += i.pt();
      return v;
    }
  };
  template <typename ColType, typename ValType>
  class MassSum {
  public:
    static ColType convert(ValType x) {
      if (x.empty())
        return 0;
      auto v = x[0].p4();
      for (const auto& i : x)
        v += i.p4();
      return v.mass();
    }
  };
  template <typename ColType, typename ValType>
  class PtVectorSum {
  public:
    static ColType convert(ValType x) {
      if (x.empty())
        return 0;
      auto v = x[0].p4();
      v -= x[0].p4();
      for (const auto& i : x)
        v += i.p4();
      return v.pt();
    }
  };

  template <typename ValType, typename ColType = ValType, typename Converter = Identity<ValType>>
  class VariableT : public Variable {
  public:
    VariableT(const std::string& aname, const edm::ParameterSet& cfg, edm::ConsumesCollector&& cc)
        : Variable(aname, cfg), src_(cc.consumes<ValType>(cfg.getParameter<edm::InputTag>("src"))) {}
    ~VariableT() override {}
    void fill(const edm::Event& iEvent, nanoaod::FlatTable& out) const override {
      out.template addColumnValue<ColType>(
          this->name_, Converter::convert(iEvent.get(src_)), this->doc_, this->precision_);
    }

  protected:
    edm::EDGetTokenT<ValType> src_;
  };
  typedef VariableT<int> IntVar;
  typedef VariableT<float> FloatVar;
  typedef VariableT<double, float> DoubleVar;
  typedef VariableT<bool> BoolVar;
  typedef VariableT<edm::View<reco::Candidate>, float, ScalarPtSum<float, edm::View<reco::Candidate>>>
      CandidateScalarSumVar;
  typedef VariableT<edm::View<reco::Candidate>, float, MassSum<float, edm::View<reco::Candidate>>> CandidateSumMassVar;
  typedef VariableT<edm::View<reco::Candidate>, int, Size<edm::View<reco::Candidate>>> CandidateSizeVar;
  std::vector<std::unique_ptr<Variable>> vars_;
  const std::string name_;
  const bool extension_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GlobalVariablesTableProducer);
