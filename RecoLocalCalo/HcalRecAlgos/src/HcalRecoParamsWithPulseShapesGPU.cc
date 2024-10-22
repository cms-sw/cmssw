#include "RecoLocalCalo/HcalRecAlgos/interface/HcalRecoParamsWithPulseShapesGPU.h"

#include "CondFormats/HcalObjects/interface/HcalRecoParams.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShapes.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/PulseShapeFunctor.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include <unordered_map>

// FIXME: add proper getters to conditions
HcalRecoParamsWithPulseShapesGPU::HcalRecoParamsWithPulseShapesGPU(HcalRecoParams const& recoParams)
    : totalChannels_{recoParams.getAllContainers()[0].second.size() + recoParams.getAllContainers()[1].second.size()},
      param1_(totalChannels_),
      param2_(totalChannels_),
      ids_(totalChannels_) {
#ifdef HCAL_MAHI_CPUDEBUG
  printf("hello from a reco params with pulse shapes\n");
#endif

  auto const containers = recoParams.getAllContainers();

  HcalPulseShapes pulseShapes;
  std::unordered_map<unsigned int, uint32_t> idCache;

  // fill in eb
  auto const& barrelValues = containers[0].second;
  for (uint64_t i = 0; i < barrelValues.size(); ++i) {
    param1_[i] = barrelValues[i].param1();
    param2_[i] = barrelValues[i].param2();

    auto const pulseShapeId = barrelValues[i].pulseShapeID();
    // FIXME: 0 throws upon look up to HcalPulseShapes
    // although comments state that 0 is reserved,
    // HcalPulseShapes::getShape throws on 0!
    if (pulseShapeId == 0) {
      ids_[i] = 0;
      continue;
    }
    if (auto const iter = idCache.find(pulseShapeId); iter == idCache.end()) {
      // new guy
      auto const newId = idCache.size();
      idCache[pulseShapeId] = newId;
      // this will be the id
      ids_[i] = newId;

      // resize value arrays
      acc25nsVec_.resize(acc25nsVec_.size() + hcal::constants::maxPSshapeBin);
      diff25nsItvlVec_.resize(diff25nsItvlVec_.size() + hcal::constants::maxPSshapeBin);
      accVarLenIdxMinusOneVec_.resize(accVarLenIdxMinusOneVec_.size() + hcal::constants::nsPerBX);
      diffVarItvlIdxMinusOneVec_.resize(diffVarItvlIdxMinusOneVec_.size() + hcal::constants::nsPerBX);
      accVarLenIdxZEROVec_.resize(accVarLenIdxZEROVec_.size() + hcal::constants::nsPerBX);
      diffVarItvlIdxZEROVec_.resize(diffVarItvlIdxZEROVec_.size() + hcal::constants::nsPerBX);

      // precompute and get values from the functor
      auto const& pulseShape = pulseShapes.getShape(pulseShapeId);
      FitterFuncs::PulseShapeFunctor functor{pulseShape, false, false, false, 1, 0, 0, hcal::constants::maxSamples};
      auto const offset256 = newId * hcal::constants::maxPSshapeBin;
      auto const offset25 = newId * hcal::constants::nsPerBX;
      auto const numShapes = newId;
      for (int i = 0; i < hcal::constants::maxPSshapeBin; i++) {
        acc25nsVec_[offset256 * numShapes + i] = functor.acc25nsVec()[i];
        diff25nsItvlVec_[offset256 * numShapes + i] = functor.diff25nsItvlVec()[i];
      }

      for (int i = 0; i < hcal::constants::nsPerBX; i++) {
        accVarLenIdxMinusOneVec_[offset25 * numShapes + i] = functor.accVarLenIdxMinusOneVec()[i];
        diffVarItvlIdxMinusOneVec_[offset25 * numShapes + i] = functor.diffVarItvlIdxMinusOneVec()[i];
        accVarLenIdxZEROVec_[offset25 * numShapes + i] = functor.accVarLenIdxZEROVec()[i];
        diffVarItvlIdxZEROVec_[offset25 * numShapes + i] = functor.diffVarItvlIdxZEROVec()[i];
      }
    } else {
      // already recorded this pulse shape, just set id
      ids_[i] = iter->second;
    }
#ifdef HCAL_MAHI_CPUDEBUG
    if (barrelValues[i].rawId() == DETID_TO_DEBUG) {
      printf("recoShapeId = %u myid = %u\n", pulseShapeId, ids_[i]);
    }
#endif
  }

  // fill in ee
  auto const& endcapValues = containers[1].second;
  auto const offset = barrelValues.size();
  for (uint64_t i = 0; i < endcapValues.size(); ++i) {
    param1_[i + offset] = endcapValues[i].param1();
    param2_[i + offset] = endcapValues[i].param2();

    auto const pulseShapeId = endcapValues[i].pulseShapeID();
    // FIXME: 0 throws upon look up to HcalPulseShapes
    // although comments state that 0 is reserved,
    // HcalPulseShapes::getShape throws on 0!
    if (pulseShapeId == 0) {
      ids_[i + offset] = 0;
      continue;
    }
    if (auto const iter = idCache.find(pulseShapeId); iter == idCache.end()) {
      // new guy
      auto const newId = idCache.size();
      idCache[pulseShapeId] = newId;
      // this will be the id
      ids_[i + offset] = newId;

      // resize value arrays
      acc25nsVec_.resize(acc25nsVec_.size() + hcal::constants::maxPSshapeBin);
      diff25nsItvlVec_.resize(diff25nsItvlVec_.size() + hcal::constants::maxPSshapeBin);
      accVarLenIdxMinusOneVec_.resize(accVarLenIdxMinusOneVec_.size() + hcal::constants::nsPerBX);
      diffVarItvlIdxMinusOneVec_.resize(diffVarItvlIdxMinusOneVec_.size() + hcal::constants::nsPerBX);
      accVarLenIdxZEROVec_.resize(accVarLenIdxZEROVec_.size() + hcal::constants::nsPerBX);
      diffVarItvlIdxZEROVec_.resize(diffVarItvlIdxZEROVec_.size() + hcal::constants::nsPerBX);

      // precompute and get values from the functor
      auto const& pulseShape = pulseShapes.getShape(pulseShapeId);
      FitterFuncs::PulseShapeFunctor functor{pulseShape, false, false, false, 1, 0, 0, hcal::constants::maxSamples};
      auto const offset256 = newId * hcal::constants::maxPSshapeBin;
      auto const offset25 = newId * hcal::constants::nsPerBX;
      auto const numShapes = newId;
      for (int i = 0; i < hcal::constants::maxPSshapeBin; i++) {
        acc25nsVec_[offset256 * numShapes + i] = functor.acc25nsVec()[i];
        diff25nsItvlVec_[offset256 * numShapes + i] = functor.diff25nsItvlVec()[i];
      }

      for (int i = 0; i < hcal::constants::nsPerBX; i++) {
        accVarLenIdxMinusOneVec_[offset25 * numShapes + i] = functor.accVarLenIdxMinusOneVec()[i];
        diffVarItvlIdxMinusOneVec_[offset25 * numShapes + i] = functor.diffVarItvlIdxMinusOneVec()[i];
        accVarLenIdxZEROVec_[offset25 * numShapes + i] = functor.accVarLenIdxZEROVec()[i];
        diffVarItvlIdxZEROVec_[offset25 * numShapes + i] = functor.diffVarItvlIdxZEROVec()[i];
      }
    } else {
      // already recorded this pulse shape, just set id
      ids_[i + offset] = iter->second;
    }
  }

#ifdef HCAL_MAHI_CPUDEBUG
  for (auto const& p : idCache)
    printf("recoPulseShapeId = %u id = %u\n", p.first, p.second);
#endif
}

HcalRecoParamsWithPulseShapesGPU::Product::~Product() {
  // deallocation
  cudaCheck(cudaFree(param1));
  cudaCheck(cudaFree(param2));
  cudaCheck(cudaFree(ids));
  cudaCheck(cudaFree(acc25nsVec));
  cudaCheck(cudaFree(diff25nsItvlVec));
  cudaCheck(cudaFree(accVarLenIdxMinusOneVec));
  cudaCheck(cudaFree(diffVarItvlIdxMinusOneVec));
  cudaCheck(cudaFree(accVarLenIdxZEROVec));
  cudaCheck(cudaFree(diffVarItvlIdxZEROVec));
}

HcalRecoParamsWithPulseShapesGPU::Product const& HcalRecoParamsWithPulseShapesGPU::getProduct(
    cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](HcalRecoParamsWithPulseShapesGPU::Product& product, cudaStream_t cudaStream) {
        // malloc
        cudaCheck(cudaMalloc((void**)&product.param1, this->param1_.size() * sizeof(uint32_t)));
        cudaCheck(cudaMalloc((void**)&product.param2, this->param2_.size() * sizeof(uint32_t)));
        cudaCheck(cudaMalloc((void**)&product.ids, this->ids_.size() * sizeof(uint32_t)));
        cudaCheck(cudaMalloc((void**)&product.acc25nsVec, this->acc25nsVec_.size() * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&product.diff25nsItvlVec, this->diff25nsItvlVec_.size() * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&product.accVarLenIdxMinusOneVec,
                             this->accVarLenIdxMinusOneVec_.size() * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&product.diffVarItvlIdxMinusOneVec,
                             this->diffVarItvlIdxMinusOneVec_.size() * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&product.accVarLenIdxZEROVec, this->accVarLenIdxZEROVec_.size() * sizeof(float)));
        cudaCheck(
            cudaMalloc((void**)&product.diffVarItvlIdxZEROVec, this->diffVarItvlIdxZEROVec_.size() * sizeof(float)));

        // transfer
        cudaCheck(cudaMemcpyAsync(product.param1,
                                  this->param1_.data(),
                                  this->param1_.size() * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.param2,
                                  this->param2_.data(),
                                  this->param2_.size() * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(
            product.ids, this->ids_.data(), this->ids_.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, cudaStream));
        cudaCheck(cudaMemcpyAsync(product.acc25nsVec,
                                  this->acc25nsVec_.data(),
                                  this->acc25nsVec_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.diff25nsItvlVec,
                                  this->diff25nsItvlVec_.data(),
                                  this->diff25nsItvlVec_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.accVarLenIdxMinusOneVec,
                                  this->accVarLenIdxMinusOneVec_.data(),
                                  this->accVarLenIdxMinusOneVec_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.diffVarItvlIdxMinusOneVec,
                                  this->diffVarItvlIdxMinusOneVec_.data(),
                                  this->diffVarItvlIdxMinusOneVec_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.accVarLenIdxZEROVec,
                                  this->accVarLenIdxZEROVec_.data(),
                                  this->accVarLenIdxZEROVec_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.diffVarItvlIdxZEROVec,
                                  this->diffVarItvlIdxZEROVec_.data(),
                                  this->diffVarItvlIdxZEROVec_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
      });

  return product;
}

TYPELOOKUP_DATA_REG(HcalRecoParamsWithPulseShapesGPU);
