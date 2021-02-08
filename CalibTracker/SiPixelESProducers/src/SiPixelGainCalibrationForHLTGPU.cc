#include <cuda.h>

#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationForHLTGPU.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationForHLT.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainForHLTonGPU.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

SiPixelGainCalibrationForHLTGPU::SiPixelGainCalibrationForHLTGPU(const SiPixelGainCalibrationForHLT& gains,
                                                                 const TrackerGeometry& geom)
    : gains_(&gains) {
  // bizzarre logic (looking for fist strip-det) don't ask
  auto const& dus = geom.detUnits();
  unsigned int n_detectors = dus.size();
  for (unsigned int i = 1; i < 7; ++i) {
    const auto offset = geom.offsetDU(GeomDetEnumerators::tkDetEnum[i]);
    if (offset != dus.size() && dus[offset]->type().isTrackerStrip()) {
      if (n_detectors > offset)
        n_detectors = offset;
    }
  }

  LogDebug("SiPixelGainCalibrationForHLTGPU")
      << "caching calibs for " << n_detectors << " pixel detectors of size " << gains.data().size() << '\n'
      << "sizes " << sizeof(char) << ' ' << sizeof(uint8_t) << ' ' << sizeof(SiPixelGainForHLTonGPU::DecodingStructure);

  cudaCheck(cudaMallocHost((void**)&gainForHLTonHost_, sizeof(SiPixelGainForHLTonGPU)));
  gainForHLTonHost_->v_pedestals_ =
      (SiPixelGainForHLTonGPU_DecodingStructure*)this->gains_->data().data();  // so it can be used on CPU as well...

  // do not read back from the (possibly write-combined) memory buffer
  auto minPed = gains.getPedLow();
  auto maxPed = gains.getPedHigh();
  auto minGain = gains.getGainLow();
  auto maxGain = gains.getGainHigh();
  auto nBinsToUseForEncoding = 253;

  // we will simplify later (not everything is needed....)
  gainForHLTonHost_->minPed_ = minPed;
  gainForHLTonHost_->maxPed_ = maxPed;
  gainForHLTonHost_->minGain_ = minGain;
  gainForHLTonHost_->maxGain_ = maxGain;

  gainForHLTonHost_->numberOfRowsAveragedOver_ = 80;
  gainForHLTonHost_->nBinsToUseForEncoding_ = nBinsToUseForEncoding;
  gainForHLTonHost_->deadFlag_ = 255;
  gainForHLTonHost_->noisyFlag_ = 254;

  gainForHLTonHost_->pedPrecision_ = static_cast<float>(maxPed - minPed) / nBinsToUseForEncoding;
  gainForHLTonHost_->gainPrecision_ = static_cast<float>(maxGain - minGain) / nBinsToUseForEncoding;

  LogDebug("SiPixelGainCalibrationForHLTGPU")
      << "precisions g " << gainForHLTonHost_->pedPrecision_ << ' ' << gainForHLTonHost_->gainPrecision_;

  // fill the index map
  auto const& ind = gains.getIndexes();
  LogDebug("SiPixelGainCalibrationForHLTGPU") << ind.size() << " " << n_detectors;

  for (auto i = 0U; i < n_detectors; ++i) {
    auto p = std::lower_bound(
        ind.begin(), ind.end(), dus[i]->geographicalId().rawId(), SiPixelGainCalibrationForHLT::StrictWeakOrdering());
    assert(p != ind.end() && p->detid == dus[i]->geographicalId());
    assert(p->iend <= gains.data().size());
    assert(p->iend >= p->ibegin);
    assert(0 == p->ibegin % 2);
    assert(0 == p->iend % 2);
    assert(p->ibegin != p->iend);
    assert(p->ncols > 0);
    gainForHLTonHost_->rangeAndCols_[i] = std::make_pair(SiPixelGainForHLTonGPU::Range(p->ibegin, p->iend), p->ncols);
    if (ind[i].detid != dus[i]->geographicalId())
      LogDebug("SiPixelGainCalibrationForHLTGPU") << ind[i].detid << "!=" << dus[i]->geographicalId();
  }
}

SiPixelGainCalibrationForHLTGPU::~SiPixelGainCalibrationForHLTGPU() { cudaCheck(cudaFreeHost(gainForHLTonHost_)); }

SiPixelGainCalibrationForHLTGPU::GPUData::~GPUData() {
  cudaCheck(cudaFree(gainForHLTonGPU));
  cudaCheck(cudaFree(gainDataOnGPU));
}

const SiPixelGainForHLTonGPU* SiPixelGainCalibrationForHLTGPU::getGPUProductAsync(cudaStream_t cudaStream) const {
  const auto& data = gpuData_.dataForCurrentDeviceAsync(cudaStream, [this](GPUData& data, cudaStream_t stream) {
    cudaCheck(cudaMalloc((void**)&data.gainForHLTonGPU, sizeof(SiPixelGainForHLTonGPU)));
    cudaCheck(cudaMalloc((void**)&data.gainDataOnGPU, this->gains_->data().size()));
    // gains.data().data() is used also for non-GPU code, we cannot allocate it on aligned and write-combined memory
    cudaCheck(cudaMemcpyAsync(
        data.gainDataOnGPU, this->gains_->data().data(), this->gains_->data().size(), cudaMemcpyDefault, stream));

    cudaCheck(cudaMemcpyAsync(
        data.gainForHLTonGPU, this->gainForHLTonHost_, sizeof(SiPixelGainForHLTonGPU), cudaMemcpyDefault, stream));
    cudaCheck(cudaMemcpyAsync(&(data.gainForHLTonGPU->v_pedestals_),
                              &(data.gainDataOnGPU),
                              sizeof(SiPixelGainForHLTonGPU_DecodingStructure*),
                              cudaMemcpyDefault,
                              stream));
  });
  return data.gainForHLTonGPU;
}
