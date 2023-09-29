#ifndef MuonTools_MuRecObjBaseProducer_h
#define MuonTools_MuRecObjBaseProducer_h

/** \class MuRecObjBaseProducer MuRecObjBaseProducer.h DPGAnalysis/MuonTools/src/MuRecObjBaseProducer.h
 *  
 * Helper class defining the generic interface of a muon digi Producer
 *
 * \author C. Battilana (INFN BO)
 *
 *
 */

#include "DataFormats/Common/interface/RangeMap.h"
#include "DataFormats/Common/interface/ClonePolicy.h"
#include "DataFormats/Common/interface/OwnVector.h"

#include "PhysicsTools/NanoAOD/interface/SimpleFlatTableProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackingRecHit/interface/RecSegment.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"

#include <algorithm>
#include <type_traits>
#include <list>
#include <string>

template <class DETECTOR_T, class RECO_T, class GEOM_T>
class MuRecObjBaseProducer
    : public SimpleFlatTableProducerBase<RECO_T, edm::RangeMap<DETECTOR_T, edm::OwnVector<RECO_T>>> {
  using COLLECTION = edm::RangeMap<DETECTOR_T, edm::OwnVector<RECO_T>>;

  edm::ESGetToken<GEOM_T, MuonGeometryRecord> m_token;
  edm::ESHandle<GEOM_T> m_geometry;

  using IntDetVar = FuncVariable<DETECTOR_T, StringObjectFunction<DETECTOR_T>, int>;
  using UIntDetVar = FuncVariable<DETECTOR_T, StringObjectFunction<DETECTOR_T>, unsigned int>;
  using Int8DetVar = FuncVariable<DETECTOR_T, StringObjectFunction<DETECTOR_T>, int8_t>;
  using UInt8DetVar = FuncVariable<DETECTOR_T, StringObjectFunction<DETECTOR_T>, uint8_t>;

  std::vector<std::unique_ptr<Variable<DETECTOR_T>>> detIdVars_;

  using GlobalPosVar = FuncVariable<GlobalPoint, StringObjectFunction<GlobalPoint>, float>;
  using GlobalDirVar = FuncVariable<GlobalVector, StringObjectFunction<GlobalVector>, float>;

  std::vector<std::unique_ptr<Variable<GlobalPoint>>> globalPosVars_;
  std::vector<std::unique_ptr<Variable<GlobalVector>>> globalDirVars_;

public:
  MuRecObjBaseProducer(edm::ParameterSet const &params)
      : SimpleFlatTableProducerBase<RECO_T, COLLECTION>(params), m_token{this->template esConsumes()} {
    auto varCfgs = params.getParameter<edm::ParameterSet>("detIdVariables");
    auto varNames = varCfgs.getParameterNamesForType<edm::ParameterSet>();

    std::transform(varNames.begin(), varNames.end(), std::back_inserter(detIdVars_), [&](const auto &name) {
      const edm::ParameterSet &varCfg = varCfgs.getParameter<edm::ParameterSet>(name);
      const std::string &type = varCfg.getParameter<std::string>("type");

      std::unique_ptr<Variable<DETECTOR_T>> detVarPtr;

      if (type == "int") {
        detVarPtr = std::move(std::make_unique<IntDetVar>(name, varCfg));  // CB can improve?
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

    varCfgs = params.getParameter<edm::ParameterSet>("globalPosVariables");
    varNames = varCfgs.getParameterNamesForType<edm::ParameterSet>();

    std::transform(varNames.begin(), varNames.end(), std::back_inserter(globalPosVars_), [&](const auto &name) {
      return std::make_unique<GlobalPosVar>(name, varCfgs.getParameter<edm::ParameterSet>(name));
    });

    if constexpr (std::is_base_of_v<RecSegment, RECO_T>) {
      varCfgs = params.getParameter<edm::ParameterSet>("globalDirVariables");
      varNames = varCfgs.getParameterNamesForType<edm::ParameterSet>();

      std::transform(varNames.begin(), varNames.end(), std::back_inserter(globalDirVars_), [&](const auto &name) {
        return std::make_unique<GlobalDirVar>(name, varCfgs.getParameter<edm::ParameterSet>(name));
      });
    }
  }

  ~MuRecObjBaseProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc = SimpleFlatTableProducerBase<RECO_T, COLLECTION>::baseDescriptions();

    auto baseDescription = []() {
      edm::ParameterSetDescription varBase;

      varBase.add<std::string>("expr")->setComment("a function to define the content of the branch in the flat table");
      varBase.add<std::string>("doc")->setComment("few words description of the branch content");

      return varBase;
    };

    auto fullDescription = [](auto const &var, std::string const label) {
      edm::ParameterSetDescription fullDesc;

      edm::ParameterWildcard<edm::ParameterSetDescription> detIdVarWildCard{"*", edm::RequireZeroOrMore, true, var};
      fullDesc.setComment("a parameters set to define all " + label + " variables to the flat table");
      fullDesc.addNode(detIdVarWildCard);

      return fullDesc;
    };

    auto detIdVar{baseDescription()};
    auto globalGeomVar{baseDescription()};

    edm::Comment comType{"the c++ type of the branch in the flat table"};
    detIdVar.ifValue(edm::ParameterDescription<std::string>{"type", "int", true, comType},
                     edm::allowedValues<std::string>("int", "uint", "int8", "uint8"));

    edm::Comment comPrecision{"the precision with which to store the value in the flat table"};
    globalGeomVar.addOptionalNode(edm::ParameterDescription<int>{"precision", true, comPrecision}, false);

    desc.add<edm::ParameterSetDescription>("detIdVariables", fullDescription(detIdVar, "DetId"));
    desc.add<edm::ParameterSetDescription>("globalPosVariables", fullDescription(globalGeomVar, "Global Position"));

    if constexpr (std::is_base_of_v<RecSegment, RECO_T>) {
      desc.add<edm::ParameterSetDescription>("globalDirVariables", fullDescription(globalGeomVar, "Global Direction"));
    }

    descriptions.addWithDefaultLabel(desc);
  }

  std::unique_ptr<nanoaod::FlatTable> fillTable(const edm::Event &iEvent,
                                                const edm::Handle<COLLECTION> &product) const override {
    std::vector<const RECO_T *> objs;
    std::vector<const DETECTOR_T *> detIds;
    std::vector<const GlobalPoint *> globalPositions;
    std::vector<const GlobalVector *> globalDirections;

    // CB needed to store DetIds, global points and vectors (they are transient)
    std::list<DETECTOR_T> detIdObjs;
    std::list<GlobalPoint> globalPointObjs;
    std::list<GlobalVector> globalVectorObjs;

    if (product.isValid()) {
      auto detIdIt = product->id_begin();
      const auto detIdEnd = product->id_end();

      for (; detIdIt != detIdEnd; ++detIdIt) {
        const auto &range = product->get(*detIdIt);
        const GeomDet *geomDet = m_geometry->idToDet(*detIdIt);

        detIdObjs.push_back(*detIdIt);
        std::fill_n(std::back_inserter(detIds), range.second - range.first, &detIdObjs.back());

        for (auto objIt{range.first}; objIt != range.second; ++objIt) {
          objs.push_back(&(*objIt));
          globalPointObjs.push_back(geomDet->toGlobal(objIt->localPosition()));
          globalPositions.push_back(&globalPointObjs.back());
          if constexpr (std::is_base_of_v<RecSegment, RECO_T>) {
            globalVectorObjs.push_back(geomDet->toGlobal(objIt->localDirection()));
            globalDirections.push_back(&globalVectorObjs.back());
          }
        }
      }
    }

    auto table = std::make_unique<nanoaod::FlatTable>(objs.size(), this->name_, false, this->extension_);

    for (const auto &var : this->vars_) {
      var->fill(objs, *table);
    }

    for (const auto &var : detIdVars_) {
      var->fill(detIds, *table);
    }

    for (const auto &var : globalPosVars_) {
      var->fill(globalPositions, *table);
    }

    if constexpr (std::is_base_of_v<RecSegment, RECO_T>) {
      for (const auto &var : globalDirVars_) {
        var->fill(globalDirections, *table);
      }
    }

    return table;
  }

  void produce(edm::Event &event, const edm::EventSetup &environment) override {
    edm::Handle<COLLECTION> src;
    event.getByToken(this->src_, src);

    m_geometry = environment.getHandle(m_token);
    std::unique_ptr<nanoaod::FlatTable> out = fillTable(event, src);
    out->setDoc(this->doc_);

    event.put(std::move(out));
  }
};

#endif
