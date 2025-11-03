#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/EmptyGroupDescription.h"
#include "FWCore/ParameterSet/interface/allowedValues.h"
#include "Utilities/General/interface/ClassName.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "PhysicsTools/NanoAOD/interface/SimpleFlatTableProducer.h"

#include "DataFormats/L1Scouting/interface/OrbitCollection.h"
#include "DataFormats/NanoAOD/interface/OrbitFlatTable.h"

template <typename T>
class SimpleOrbitFlatTableProducer : public edm::stream::EDProducer<> {
public:
  using TOrbitCollection = OrbitCollection<T>;

  SimpleOrbitFlatTableProducer(edm::ParameterSet const &params)
      : name_(params.getParameter<std::string>("name")),
        doc_(params.getParameter<std::string>("doc")),
        singleton_(params.getParameter<bool>("singleton")),
        extension_(params.getParameter<bool>("extension")),
        skipNonExistingSrc_(params.getParameter<bool>("skipNonExistingSrc")),
        cut_(!singleton_ ? params.getParameter<std::string>("cut") : "",
             !singleton_ && params.existsAs<bool>("lazyEval") ? params.getUntrackedParameter<bool>("lazyEval") : false),
        maxLen_(params.existsAs<unsigned int>("maxLen") ? params.getParameter<unsigned int>("maxLen")
                                                        : std::numeric_limits<unsigned int>::max()),
        src_(consumes(params.getParameter<edm::InputTag>("src"))) {
    // variables
    edm::ParameterSet const &varsPSet = params.getParameter<edm::ParameterSet>("variables");
    for (const std::string &vname : varsPSet.getParameterNamesForType<edm::ParameterSet>()) {
      const auto &varPSet = varsPSet.getParameter<edm::ParameterSet>(vname);
      const std::string &type = varPSet.getParameter<std::string>("type");
      if (type == "int")
        vars_.push_back(std::make_unique<IntVar>(vname, varPSet));
      else if (type == "uint")
        vars_.push_back(std::make_unique<UIntVar>(vname, varPSet));
      else if (type == "int64")
        vars_.push_back(std::make_unique<Int64Var>(vname, varPSet));
      else if (type == "uint64")
        vars_.push_back(std::make_unique<UInt64Var>(vname, varPSet));
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

    // external variables
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
        else if (type == "int64")
          extvars_.push_back(
              std::make_unique<Int64ExtVar>(vname, varPSet, this->consumesCollector(), this->skipNonExistingSrc_));
        else if (type == "uint64")
          extvars_.push_back(
              std::make_unique<UInt64ExtVar>(vname, varPSet, this->consumesCollector(), this->skipNonExistingSrc_));
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

    produces<l1ScoutingRun3::OrbitFlatTable>();
  }

  ~SimpleOrbitFlatTableProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;
    const std::string& classname = ClassName<T>::name();
    desc.add<std::string>("name")->setComment("name of the branch in the flat table output for " + classname);
    desc.add<std::string>("doc", "")->setComment("few words of self documentation");
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
        "define the maximum length per bx of the input collection to put in the branch");
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

    // external variables
    edm::ParameterSetDescription extvariable;
    extvariable.add<edm::InputTag>("src")->setComment("valuemap input collection to fill the flat table");
    extvariable.add<std::string>("doc")->setComment("few words description of the branch content");
    extvariable.ifValue(edm::ParameterDescription<std::string>(
                            "type", "int", true, edm::Comment("the c++ type of the branch in the flat table")),
                        edm::allowedValues<std::string>(
                            "int", "uint", "int64", "uint64", "float", "double", "uint8", "int16", "uint16", "bool"));
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

    descriptions.addWithDefaultLabel(desc);
  }

  void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) override {
    edm::Handle<TOrbitCollection> src;
    iEvent.getByToken(src_, src);

    std::vector<const T *> selobjs;
    std::vector<edm::Ptr<T>> selptrs;  // for external variables
    std::vector<unsigned int> selbxOffsets;

    if (src.isValid() || !skipNonExistingSrc_) {
      if (singleton_) {
        // fill objects
        for (int i = 0; i < src->size(); i++) {
          selobjs.push_back(&(*src)[i]);
          if (!extvars_.empty())
            selptrs.emplace_back(src, 0);
        }

        // set bx offsets of selected objects
        selbxOffsets = src->bxOffsets();

      } else {  // not singleton
        // offsets before cut
        std::vector<unsigned int> bxOffsets = src->bxOffsets();

        // number of objects per bx after cut
        std::vector<unsigned int> selbxSizes = std::vector<unsigned int>(l1ScoutingRun3::OrbitFlatTable::NBX + 1, 0);

        for (const unsigned int &bx : src->getFilledBxs()) {
          const auto &objs = src->bxIterator(bx);
          for (unsigned int i = 0; i < objs.size(); i++) {
            const auto &obj = objs[i];
            edm::Ptr<T> objptr = edm::Ptr<T>(src, bxOffsets[bx] + i);
            if (cut_(obj)) {  // apply cut
              selobjs.push_back(&obj);
              if (!extvars_.empty())
                selptrs.push_back(objptr);
              selbxSizes[bx]++;
            }
            if (selbxSizes[bx] >= maxLen_)  // skip to the next bx
              break;
          }
        }

        selbxOffsets = sizes2offsets(selbxSizes);
      }
    }

    auto out = std::make_unique<l1ScoutingRun3::OrbitFlatTable>(selbxOffsets, name_, singleton_, extension_);
    out->setDoc(doc_);

    for (const auto &var : this->vars_)
      var->fill(selobjs, *out);
    for (const auto &var : this->extvars_)
      var->fill(iEvent, selptrs, *out);

    iEvent.put(std::move(out));
  }

private:  // private attributes
  const std::string name_;
  const std::string doc_;
  const bool singleton_;
  const bool extension_;
  const bool skipNonExistingSrc_;
  const StringCutObjectSelector<T> cut_;
  const unsigned int maxLen_;
  const edm::EDGetTokenT<TOrbitCollection> src_;

  // variables
  typedef FuncVariable<T, StringObjectFunction<T>, int32_t> IntVar;
  typedef FuncVariable<T, StringObjectFunction<T>, uint32_t> UIntVar;
  typedef FuncVariable<T, StringObjectFunction<T>, int64_t> Int64Var;
  typedef FuncVariable<T, StringObjectFunction<T>, uint64_t> UInt64Var;
  typedef FuncVariable<T, StringObjectFunction<T>, float> FloatVar;
  typedef FuncVariable<T, StringObjectFunction<T>, double> DoubleVar;
  typedef FuncVariable<T, StringObjectFunction<T>, uint8_t> UInt8Var;
  typedef FuncVariable<T, StringObjectFunction<T>, int16_t> Int16Var;
  typedef FuncVariable<T, StringObjectFunction<T>, uint16_t> UInt16Var;
  typedef FuncVariable<T, StringCutObjectSelector<T>, bool> BoolVar;
  std::vector<std::unique_ptr<Variable<T>>> vars_;

  // external variables
  typedef ValueMapVariable<T, int32_t> IntExtVar;
  typedef ValueMapVariable<T, uint32_t> UIntExtVar;
  typedef ValueMapVariable<T, int64_t> Int64ExtVar;
  typedef ValueMapVariable<T, uint64_t> UInt64ExtVar;
  typedef ValueMapVariable<T, float> FloatExtVar;
  typedef ValueMapVariable<T, double, float> DoubleExtVar;
  typedef ValueMapVariable<T, bool> BoolExtVar;
  typedef ValueMapVariable<T, int, uint8_t> UInt8ExtVar;
  typedef ValueMapVariable<T, int, int16_t> Int16ExtVar;
  typedef ValueMapVariable<T, int, uint16_t> UInt16ExtVar;
  std::vector<std::unique_ptr<ExtVariable<T>>> extvars_;

private:  // private methods
  std::vector<unsigned int> sizes2offsets(std::vector<unsigned int> sizes) const {
    std::vector<unsigned int> offsets(sizes.size() + 1, 0);  // add extra one for the last offset = total size
    for (unsigned int i = 0; i < sizes.size(); i++) {
      offsets[i + 1] = offsets[i] + sizes[i];
    }
    return offsets;
  }
};

#include "DataFormats/L1Scouting/interface/L1ScoutingMuon.h"
typedef SimpleOrbitFlatTableProducer<l1ScoutingRun3::Muon> SimpleL1ScoutingMuonOrbitFlatTableProducer;

#include "DataFormats/L1Scouting/interface/L1ScoutingCalo.h"
typedef SimpleOrbitFlatTableProducer<l1ScoutingRun3::EGamma> SimpleL1ScoutingEGammaOrbitFlatTableProducer;
typedef SimpleOrbitFlatTableProducer<l1ScoutingRun3::Tau> SimpleL1ScoutingTauOrbitFlatTableProducer;
typedef SimpleOrbitFlatTableProducer<l1ScoutingRun3::Jet> SimpleL1ScoutingJetOrbitFlatTableProducer;

#include "DataFormats/L1Scouting/interface/L1ScoutingBMTFStub.h"
typedef SimpleOrbitFlatTableProducer<l1ScoutingRun3::BMTFStub> SimpleL1ScoutingBMTFStubOrbitFlatTableProducer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SimpleL1ScoutingMuonOrbitFlatTableProducer);
DEFINE_FWK_MODULE(SimpleL1ScoutingEGammaOrbitFlatTableProducer);
DEFINE_FWK_MODULE(SimpleL1ScoutingTauOrbitFlatTableProducer);
DEFINE_FWK_MODULE(SimpleL1ScoutingJetOrbitFlatTableProducer);
DEFINE_FWK_MODULE(SimpleL1ScoutingBMTFStubOrbitFlatTableProducer);
