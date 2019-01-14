#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_SiPixelRawToClusterGPUKernel_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_SiPixelRawToClusterGPUKernel_h

#include <algorithm>
#include <cuda_runtime.h>
#include "cuda/api_wrappers.h"

#include "CUDADataFormats/Common/interface/host_unique_ptr.h"
#include "FWCore/Utilities/interface/typedefs.h"
#include "HeterogeneousCore/CUDAUtilities/interface/GPUSimpleVector.h"
#include "siPixelRawToClusterHeterogeneousProduct.h"

class SiPixelFedCablingMapGPU;
class SiPixelGainForHLTonGPU;

namespace pixelgpudetails {

  // Phase 1 geometry constants
  const uint32_t layerStartBit    = 20;
  const uint32_t ladderStartBit   = 12;
  const uint32_t moduleStartBit   = 2;

  const uint32_t panelStartBit    = 10;
  const uint32_t diskStartBit     = 18;
  const uint32_t bladeStartBit    = 12;

  const uint32_t layerMask        = 0xF;
  const uint32_t ladderMask       = 0xFF;
  const uint32_t moduleMask       = 0x3FF;
  const uint32_t panelMask        = 0x3;
  const uint32_t diskMask         = 0xF;
  const uint32_t bladeMask        = 0x3F;

  const uint32_t LINK_bits        = 6;
  const uint32_t ROC_bits         = 5;
  const uint32_t DCOL_bits        = 5;
  const uint32_t PXID_bits        = 8;
  const uint32_t ADC_bits         = 8;

  // special for layer 1
  const uint32_t LINK_bits_l1     = 6;
  const uint32_t ROC_bits_l1      = 5;
  const uint32_t COL_bits_l1      = 6;
  const uint32_t ROW_bits_l1      = 7;
  const uint32_t OMIT_ERR_bits    = 1;

  const uint32_t maxROCIndex      = 8;
  const uint32_t numRowsInRoc     = 80;
  const uint32_t numColsInRoc     = 52;

  const uint32_t MAX_WORD = 2000;

  const uint32_t ADC_shift  = 0;
  const uint32_t PXID_shift = ADC_shift + ADC_bits;
  const uint32_t DCOL_shift = PXID_shift + PXID_bits;
  const uint32_t ROC_shift  = DCOL_shift + DCOL_bits;
  const uint32_t LINK_shift = ROC_shift + ROC_bits_l1;
  // special for layer 1 ROC
  const uint32_t ROW_shift = ADC_shift + ADC_bits;
  const uint32_t COL_shift = ROW_shift + ROW_bits_l1;
  const uint32_t OMIT_ERR_shift = 20;

  const uint32_t LINK_mask = ~(~uint32_t(0) << LINK_bits_l1);
  const uint32_t ROC_mask  = ~(~uint32_t(0) << ROC_bits_l1);
  const uint32_t COL_mask  = ~(~uint32_t(0) << COL_bits_l1);
  const uint32_t ROW_mask  = ~(~uint32_t(0) << ROW_bits_l1);
  const uint32_t DCOL_mask = ~(~uint32_t(0) << DCOL_bits);
  const uint32_t PXID_mask = ~(~uint32_t(0) << PXID_bits);
  const uint32_t ADC_mask  = ~(~uint32_t(0) << ADC_bits);
  const uint32_t ERROR_mask = ~(~uint32_t(0) << ROC_bits_l1);
  const uint32_t OMIT_ERR_mask = ~(~uint32_t(0) << OMIT_ERR_bits);

  struct DetIdGPU {
    uint32_t RawId;
    uint32_t rocInDet;
    uint32_t moduleId;
  };

  struct Pixel {
   uint32_t row;
   uint32_t col;
  };

  class Packing {
  public:
    using PackedDigiType = uint32_t;

    // Constructor: pre-computes masks and shifts from field widths
    __host__ __device__
    inline
    constexpr Packing(unsigned int row_w, unsigned int column_w,
                      unsigned int time_w, unsigned int adc_w) :
      row_width(row_w),
      column_width(column_w),
      adc_width(adc_w),
      row_shift(0),
      column_shift(row_shift + row_w),
      time_shift(column_shift + column_w),
      adc_shift(time_shift + time_w),
      row_mask(~(~0U << row_w)),
      column_mask( ~(~0U << column_w)),
      time_mask(~(~0U << time_w)),
      adc_mask(~(~0U << adc_w)),
      rowcol_mask(~(~0U << (column_w+row_w))),
      max_row(row_mask),
      max_column(column_mask),
      max_adc(adc_mask)
    { }

    uint32_t  row_width;
    uint32_t  column_width;
    uint32_t  adc_width;

    uint32_t  row_shift;
    uint32_t  column_shift;
    uint32_t  time_shift;
    uint32_t  adc_shift;

    PackedDigiType row_mask;
    PackedDigiType column_mask;
    PackedDigiType time_mask;
    PackedDigiType adc_mask;
    PackedDigiType rowcol_mask;

    uint32_t  max_row;
    uint32_t  max_column;
    uint32_t  max_adc;
  };

  __host__ __device__
  inline
  constexpr Packing packing() {
    return Packing(11, 11, 0, 10);
  }


  __host__ __device__
  inline
  uint32_t pack(uint32_t row, uint32_t col, uint32_t adc) {
    constexpr Packing thePacking = packing();
    adc = std::min(adc, thePacking.max_adc);

    return (row << thePacking.row_shift) |
           (col << thePacking.column_shift) |
           (adc << thePacking.adc_shift);
  }

  constexpr
  uint32_t pixelToChannel( int row, int col) {
    constexpr Packing thePacking = packing();
    return (row << thePacking.column_width) | col;
  }


  using error_obj = siPixelRawToClusterHeterogeneousProduct::error_obj;


  class SiPixelRawToClusterGPUKernel {
  public:

    using GPUProduct = siPixelRawToClusterHeterogeneousProduct::GPUProduct;

    struct CPUData {
      CPUData() = default;
      ~CPUData() = default;

      CPUData(const CPUData&) = delete;
      CPUData& operator=(const CPUData&) = delete;
      CPUData(CPUData&&) = default;
      CPUData& operator=(CPUData&&) = default;
      
      edm::cuda::host::unique_ptr<uint32_t[]> nModules_Clusters; // These should really be part of the GPU product

      edm::cuda::host::unique_ptr<pixelgpudetails::error_obj[]> data;
      edm::cuda::host::unique_ptr<GPU::SimpleVector<pixelgpudetails::error_obj>> error;

      edm::cuda::host::unique_ptr<uint32_t[]> pdigi;
      edm::cuda::host::unique_ptr<uint32_t[]> rawIdArr;
      edm::cuda::host::unique_ptr<uint16_t[]> adc;
      edm::cuda::host::unique_ptr<int32_t[]> clus;
    };

    class WordFedAppender {
    public:
      WordFedAppender(cuda::stream_t<>& cudaStream);
      ~WordFedAppender() = default;

      void initializeWordFed(int fedId, unsigned int wordCounterGPU, const cms_uint32_t *src, unsigned int length);

      const unsigned int *word() const { return word_.get(); }
      const unsigned char *fedId() const { return fedId_.get(); }

    private:
      edm::cuda::host::unique_ptr<unsigned int[]> word_;
      edm::cuda::host::unique_ptr<unsigned char[]> fedId_;
    };

    SiPixelRawToClusterGPUKernel() = default;
    ~SiPixelRawToClusterGPUKernel() = default;


    SiPixelRawToClusterGPUKernel(const SiPixelRawToClusterGPUKernel&) = delete;
    SiPixelRawToClusterGPUKernel(SiPixelRawToClusterGPUKernel&&) = delete;
    SiPixelRawToClusterGPUKernel& operator=(const SiPixelRawToClusterGPUKernel&) = delete;
    SiPixelRawToClusterGPUKernel& operator=(SiPixelRawToClusterGPUKernel&&) = delete;

    void makeClustersAsync(const SiPixelFedCablingMapGPU *cablingMap, const unsigned char *modToUnp,
                           const SiPixelGainForHLTonGPU *gains,
                           const WordFedAppender& wordFed,
                           const uint32_t wordCounter, const uint32_t fedCounter, bool convertADCtoElectrons,
                           bool useQualityInfo, bool includeErrors, bool transferToCPU_, bool debug,
                           cuda::stream_t<>& stream);

    siPixelRawToClusterHeterogeneousProduct::GPUProduct getProduct() {
      return siPixelRawToClusterHeterogeneousProduct::GPUProduct(
        std::move(digis_d), std::move(clusters_d),
        nDigis,
        digis_clusters_h.nModules_Clusters[0],
        digis_clusters_h.nModules_Clusters[1]
      );
    }

    CPUData&& getCPUData() {
      // Set the vector data pointer to point to CPU
      digis_clusters_h.error->set_data(digis_clusters_h.data.get());
      return std::move(digis_clusters_h);
    }

  private:
    uint32_t nDigis = 0;

    // CPU data
    CPUData digis_clusters_h;

    // Data to be put in the event
    SiPixelDigisCUDA digis_d;
    SiPixelClustersCUDA clusters_d;
  };

  // configuration and memory buffers alocated on the GPU
  struct context {
    uint32_t * word_d;
    uint8_t *  fedId_d;
    uint32_t * pdigi_d;
    uint16_t * xx_d;
    uint16_t * yy_d;
    uint16_t * adc_d;
    uint16_t * moduleInd_d;
    uint32_t * rawIdArr_d;

    GPU::SimpleVector<error_obj> * error_d;
    error_obj * data_d;

    // these are for the clusterizer (to be moved)
    uint32_t * moduleStart_d;
    int32_t *  clus_d;
    uint32_t * clusInModule_d;
    uint32_t * moduleId_d;
    uint32_t * debug_d;
  };

  // void initCablingMap();
  context initDeviceMemory();
  void freeMemory(context &);

  // see RecoLocalTracker/SiPixelClusterizer
  // all are runtime const, should be specified in python _cfg.py
  struct ADCThreshold {
    const int     thePixelThreshold       = 1000;     // default Pixel threshold in electrons
    const int     theSeedThreshold        = 1000;     // seed thershold in electrons not used in our algo
    const float   theClusterThreshold     = 4000;     // cluster threshold in electron
    const int     ConversionFactor        =   65;     // adc to electron conversion factor

    const int     theStackADC_            =  255;     // the maximum adc count for stack layer
    const int     theFirstStack_          =    5;     // the index of the fits stack layer
    const double  theElectronPerADCGain_  =  600;     // ADC to electron conversion
  };

}

#endif // RecoLocalTracker_SiPixelClusterizer_plugins_SiPixelRawToClusterGPUKernel_h
