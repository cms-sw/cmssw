#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationForHLTGPU.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationForHLT.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainForHLTonGPU.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include <cuda.h>

SiPixelGainCalibrationForHLTGPU::SiPixelGainCalibrationForHLTGPU(const SiPixelGainCalibrationForHLT& gains, const TrackerGeometry& geom):
  gains_(&gains)
{
  // bizzarre logic (looking for fist strip-det) don't ask
  auto const & dus = geom.detUnits();
  unsigned m_detectors = dus.size();
  for(unsigned int i=1;i<7;++i) {
    if(geom.offsetDU(GeomDetEnumerators::tkDetEnum[i]) != dus.size() &&
        dus[geom.offsetDU(GeomDetEnumerators::tkDetEnum[i])]->type().isTrackerStrip()) {
      if(geom.offsetDU(GeomDetEnumerators::tkDetEnum[i]) < m_detectors) m_detectors = geom.offsetDU(GeomDetEnumerators::tkDetEnum[i]);
    }
  }

  /*
  std::cout << "caching calibs for " << m_detectors << " pixel detectors of size " << gains.data().size() << std::endl;
  std::cout << "sizes " << sizeof(char) << ' ' << sizeof(uint8_t) << ' ' << sizeof(SiPixelGainForHLTonGPU::DecodingStructure) << std::endl;
  */

  cudaCheck(cudaMallocHost((void**) & gainForHLTonHost_, sizeof(SiPixelGainForHLTonGPU)));
  //gainForHLTonHost_->v_pedestals = gainDataOnGPU_; // how to do this?

  // do not read back from the (possibly write-combined) memory buffer
  auto minPed  = gains.getPedLow();
  auto maxPed  = gains.getPedHigh();
  auto minGain = gains.getGainLow();
  auto maxGain = gains.getGainHigh();
  auto nBinsToUseForEncoding = 253;

  // we will simplify later (not everything is needed....)
  gainForHLTonHost_->minPed_ = minPed;
  gainForHLTonHost_->maxPed_ = maxPed;
  gainForHLTonHost_->minGain_= minGain;
  gainForHLTonHost_->maxGain_= maxGain;

  gainForHLTonHost_->numberOfRowsAveragedOver_ = 80;
  gainForHLTonHost_->nBinsToUseForEncoding_    = nBinsToUseForEncoding;
  gainForHLTonHost_->deadFlag_                 = 255;
  gainForHLTonHost_->noisyFlag_                = 254;

  gainForHLTonHost_->pedPrecision  = static_cast<float>(maxPed - minPed) / nBinsToUseForEncoding;
  gainForHLTonHost_->gainPrecision = static_cast<float>(maxGain - minGain) / nBinsToUseForEncoding;

  /*
  std::cout << "precisions g " << gainForHLTonHost_->pedPrecision << ' ' << gainForHLTonHost_->gainPrecision << std::endl;
  */

  // fill the index map
  auto const & ind = gains.getIndexes();
  /*
  std::cout << ind.size() << " " << m_detectors << std::endl;
  */

  for (auto i=0U; i<m_detectors; ++i) {
    auto p = std::lower_bound(ind.begin(),ind.end(),dus[i]->geographicalId().rawId(),SiPixelGainCalibrationForHLT::StrictWeakOrdering());
    assert (p!=ind.end() && p->detid==dus[i]->geographicalId());
    assert(p->iend<=gains.data().size());
    assert(p->iend>=p->ibegin);
    assert(0==p->ibegin%2);
    assert(0==p->iend%2);
    assert(p->ibegin!=p->iend);
    assert(p->ncols>0);
    gainForHLTonHost_->rangeAndCols[i] = std::make_pair(SiPixelGainForHLTonGPU::Range(p->ibegin,p->iend), p->ncols);
    // if (ind[i].detid!=dus[i]->geographicalId()) std::cout << ind[i].detid<<"!="<<dus[i]->geographicalId() << std::endl;
    // gainForHLTonHost_->rangeAndCols[i] = std::make_pair(SiPixelGainForHLTonGPU::Range(ind[i].ibegin,ind[i].iend), ind[i].ncols);
  }

}

SiPixelGainCalibrationForHLTGPU::~SiPixelGainCalibrationForHLTGPU() {
  cudaCheck(cudaFreeHost(gainForHLTonHost_));
}

SiPixelGainCalibrationForHLTGPU::GPUData::~GPUData() {
  cudaCheck(cudaFree(gainForHLTonGPU));
  cudaCheck(cudaFree(gainDataOnGPU));
}

const SiPixelGainForHLTonGPU *SiPixelGainCalibrationForHLTGPU::getGPUProductAsync(cuda::stream_t<>& cudaStream) const {
  const auto& data = gpuData_.dataForCurrentDeviceAsync(cudaStream, [this](GPUData& data, cuda::stream_t<>& stream) {
      cudaCheck(cudaMalloc((void**) & data.gainForHLTonGPU, sizeof(SiPixelGainForHLTonGPU)));
      cudaCheck(cudaMalloc((void**) & data.gainDataOnGPU, this->gains_->data().size())); // TODO: this could be changed to cuda::memory::device::unique_ptr<>
      // gains.data().data() is used also for non-GPU code, we cannot allocate it on aligned and write-combined memory
      cudaCheck(cudaMemcpyAsync(data.gainDataOnGPU, this->gains_->data().data(), this->gains_->data().size(), cudaMemcpyDefault, stream.id()));

      cudaCheck(cudaMemcpyAsync(data.gainForHLTonGPU, this->gainForHLTonHost_, sizeof(SiPixelGainForHLTonGPU), cudaMemcpyDefault, stream.id()));
      cudaCheck(cudaMemcpyAsync(&(data.gainForHLTonGPU->v_pedestals), &(data.gainDataOnGPU), sizeof(SiPixelGainForHLTonGPU_DecodingStructure*), cudaMemcpyDefault, stream.id()));
    });
  return data.gainForHLTonGPU;
}
