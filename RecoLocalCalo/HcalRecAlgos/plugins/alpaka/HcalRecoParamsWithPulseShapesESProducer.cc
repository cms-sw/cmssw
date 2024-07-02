#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/HcalObjects/interface/HcalRecoParamWithPulseShapeHost.h"
#include "CondFormats/HcalObjects/interface/HcalRecoParamWithPulseShapeSoA.h"
#include "CondFormats/DataRecord/interface/HcalRecoParamsRcd.h"

#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShapes.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/PulseShapeFunctor.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalConstants.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class HcalRecoParamWithPulseShapeESProducer : public ESProducer {
  public:
    HcalRecoParamWithPulseShapeESProducer(edm::ParameterSet const& iConfig) : ESProducer(iConfig) {
      auto cc = setWhatProduced(this);
      recoParamsToken_ = cc.consumes();
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      descriptions.addWithDefaultLabel(desc);
    }

    std::unique_ptr<hcal::HcalRecoParamWithPulseShapeHost> produce(HcalRecoParamsRcd const& iRecord) {
      auto const& recoParams = iRecord.get(recoParamsToken_);

      auto const containers = recoParams.getAllContainers();
      size_t const totalChannels =
          recoParams.getAllContainers()[0].second.size() + recoParams.getAllContainers()[1].second.size();

      //Get unique ids
      HcalPulseShapes pulseShapes;
      std::unordered_map<unsigned int, uint32_t> idCache;  //<pulseShapeId,pulseShapeIdx>

      auto const& barrelValues = containers[0].second;
      for (uint64_t i = 0; i < barrelValues.size(); ++i) {
        auto const pulseShapeId = barrelValues[i].pulseShapeID();
        if (pulseShapeId == 0)
          continue;
        if (auto const iter = idCache.find(pulseShapeId); iter == idCache.end()) {
          idCache[pulseShapeId] = idCache.size();
        }
      }
      auto const& endcapValues = containers[1].second;
      for (uint64_t i = 0; i < endcapValues.size(); ++i) {
        auto const pulseShapeId = endcapValues[i].pulseShapeID();
        if (pulseShapeId == 0)
          continue;
        if (auto const iter = idCache.find(pulseShapeId); iter == idCache.end()) {
          idCache[pulseShapeId] = idCache.size();
        }
      }

      // Fill products
      auto product = std::make_unique<hcal::HcalRecoParamWithPulseShapeHost>(
          totalChannels, idCache.size(), cms::alpakatools::host());
      auto recoView = product->recoParamView();
      auto pulseShapeView = product->pulseShapeView();
      for (uint64_t i = 0; i < barrelValues.size(); ++i) {
        auto vi = recoView[i];
        vi.param1() = barrelValues[i].param1();
        vi.param2() = barrelValues[i].param2();
        vi.ids() = (barrelValues[i].pulseShapeID() == 0)
                       ? 0
                       : idCache.at(barrelValues[i].pulseShapeID());  //idx of the pulseShape of channel i
      }
      // fill in endcap
      auto const offset = barrelValues.size();
      for (uint64_t i = 0; i < endcapValues.size(); ++i) {
        auto vi = recoView[i + offset];
        vi.param1() = endcapValues[i].param1();
        vi.param2() = endcapValues[i].param2();
        vi.ids() = (endcapValues[i].pulseShapeID() == 0)
                       ? 0
                       : idCache.at(endcapValues[i].pulseShapeID());  //idx of the pulseShape of channel i
      }

      //fill pulseShape views
      for (auto& it : idCache) {
        auto const pulseShapeId = it.first;
        auto const arrId = it.second;
        auto const& pulseShape = pulseShapes.getShape(pulseShapeId);
        FitterFuncs::PulseShapeFunctor functor{pulseShape, false, false, false, 1, 0, 0, hcal::constants::maxSamples};

        for (int i = 0; i < hcal::constants::maxPSshapeBin; i++) {
          pulseShapeView[arrId].acc25nsVec()[i] = functor.acc25nsVec()[i];
          pulseShapeView[arrId].diff25nsItvlVec()[i] = functor.diff25nsItvlVec()[i];
        }
        for (int i = 0; i < hcal::constants::nsPerBX; i++) {
          pulseShapeView[arrId].accVarLenIdxMinusOneVec()[i] = functor.accVarLenIdxMinusOneVec()[i];
          pulseShapeView[arrId].diffVarItvlIdxMinusOneVec()[i] = functor.diffVarItvlIdxMinusOneVec()[i];
          pulseShapeView[arrId].accVarLenIdxZEROVec()[i] = functor.accVarLenIdxZEROVec()[i];
          pulseShapeView[arrId].diffVarItvlIdxZEROVec()[i] = functor.diffVarItvlIdxZEROVec()[i];
        }
      }

      return product;
    }

  private:
    edm::ESGetToken<HcalRecoParams, HcalRecoParamsRcd> recoParamsToken_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(HcalRecoParamWithPulseShapeESProducer);
