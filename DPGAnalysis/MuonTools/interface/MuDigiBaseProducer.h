#ifndef MuonTools_MuDigiBaseProducer_h
#define MuonTools_MuDigiBaseProducer_h

/** \class MuDigiBaseProducer MuDigiBaseProducer.h DPGAnalysis/MuonTools/src/MuDigiBaseProducer.h
 *  
 * Helper class defining the generic interface of a muon digi Producer
 *
 * \author C. Battilana (INFN BO)
 *
 *
 */

#include "DataFormats/MuonData/interface/MuonDigiCollection.h"
#include "PhysicsTools/NanoAOD/interface/SimpleFlatTableProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <algorithm>
#include <list>
#include <string>

template <class DETECTOR_T, class DIGI_T>
class MuDigiBaseProducer : public SimpleFlatTableProducerBase<DIGI_T, MuonDigiCollection<DETECTOR_T, DIGI_T>> {
  using COLLECTION = MuonDigiCollection<DETECTOR_T, DIGI_T>;

  using IntDetVar = FuncVariable<DETECTOR_T, StringObjectFunction<DETECTOR_T>, int>;
  using UIntDetVar = FuncVariable<DETECTOR_T, StringObjectFunction<DETECTOR_T>, unsigned int>;
  using Int8DetVar = FuncVariable<DETECTOR_T, StringObjectFunction<DETECTOR_T>, int8_t>;
  using UInt8DetVar = FuncVariable<DETECTOR_T, StringObjectFunction<DETECTOR_T>, uint8_t>;

  std::vector<std::unique_ptr<Variable<DETECTOR_T>>> detIdVars_;

public:
  MuDigiBaseProducer(edm::ParameterSet const &params) : SimpleFlatTableProducerBase<DIGI_T, COLLECTION>(params) {
    const auto &varCfgs = params.getParameter<edm::ParameterSet>("detIdVariables");
    const auto &varNames = varCfgs.getParameterNamesForType<edm::ParameterSet>();

    std::transform(varNames.begin(), varNames.end(), std::back_inserter(detIdVars_), [&](const auto &name) {
      const edm::ParameterSet &varCfg = varCfgs.getParameter<edm::ParameterSet>(name);
      const std::string &type = varCfg.getParameter<std::string>("type");

      std::unique_ptr<Variable<DETECTOR_T>> detVarPtr;

      if (type == "int") {
        detVarPtr = std::move(std::make_unique<IntDetVar>(name, varCfg));
      } else if (type == "uint") {
        detVarPtr = std::move(std::make_unique<UIntDetVar>(name, varCfg));
      } else if (type == "int8") {
        detVarPtr = std::move(std::make_unique<Int8DetVar>(name, varCfg));
      } else if (type == "uint8") {
        detVarPtr = std::move(std::make_unique<UInt8DetVar>(name, varCfg));
      } else {
        throw cms::Exception("Configuration", "unsupported type " + type + " for variable " + name);
      }

      return detVarPtr;
    });
  }

  ~MuDigiBaseProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc = SimpleFlatTableProducerBase<DIGI_T, COLLECTION>::baseDescriptions();

    edm::ParameterSetDescription variable;
    edm::Comment comType{"the c++ type of the branch in the flat table"};
    edm::Comment comPrecision{"the precision with which to store the value in the flat table"};

    variable.add<std::string>("expr")->setComment("a function to define the content of the branch in the flat table");
    variable.add<std::string>("doc")->setComment("few words description of the branch content");

    variable.ifValue(edm::ParameterDescription<std::string>("type", "int", true, comType),
                     edm::allowedValues<std::string>("int", "uint", "int8", "uint8"));

    edm::ParameterSetDescription variables;

    variables.setComment("a parameters set to define all variable taken form detId to fill the flat table");

    edm::ParameterWildcard<edm::ParameterSetDescription> variableWildCard{"*", edm::RequireZeroOrMore, true, variable};
    variables.addNode(variableWildCard);

    desc.add<edm::ParameterSetDescription>("detIdVariables", variables);

    descriptions.addWithDefaultLabel(desc);
  }

  std::unique_ptr<nanoaod::FlatTable> fillTable(const edm::Event &iEvent,
                                                const edm::Handle<COLLECTION> &prod) const override {
    std::vector<const DIGI_T *> digis;
    std::vector<const DETECTOR_T *> detIds;
    std::list<DETECTOR_T> detIdObjs;  // CB needed to store DetIds (they are transient)

    if (prod.isValid()) {
      auto detIdIt = prod->begin();
      auto detIdEnd = prod->end();

      for (; detIdIt != detIdEnd; ++detIdIt) {
        const auto &[detId, range] = (*detIdIt);
        detIdObjs.push_back(detId);
        std::fill_n(std::back_inserter(detIds), range.second - range.first, &detIdObjs.back());
        std::transform(range.first, range.second, std::back_inserter(digis), [](const auto &digi) { return &digi; });
      }
    }

    auto table = std::make_unique<nanoaod::FlatTable>(digis.size(), this->name_, false, this->extension_);

    for (const auto &var : this->vars_) {
      var->fill(digis, *table);
    }

    for (const auto &var : detIdVars_) {
      var->fill(detIds, *table);
    }

    return table;
  }
};

#endif
