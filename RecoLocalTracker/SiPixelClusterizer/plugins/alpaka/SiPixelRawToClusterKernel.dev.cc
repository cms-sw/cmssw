// C++ includes
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <type_traits>

// Alpaka includes
#include <alpaka/alpaka.hpp>

// CMSSW includes
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationForHLTLayout.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelMappingLayout.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelClusterSoA/interface/ClusteringConstants.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersSoA.h"
#include "DataFormats/SiPixelClusterSoA/interface/alpaka/SiPixelClustersSoACollection.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDigi/interface/SiPixelDigiConstants.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigiErrorsSoA.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisSoA.h"
#include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigiErrorsSoACollection.h"
#include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigisSoACollection.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelErrorCompact.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/warpsize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelClusterThresholds.h"

// local includes
#include "CalibPixel.h"
#include "ClusterChargeCut.h"
#include "PixelClustering.h"
#include "SiPixelRawToClusterKernel.h"

// #define GPU_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace pixelDetails {

    ALPAKA_FN_ACC bool isBarrel(uint32_t rawId) {
      return (PixelSubdetector::PixelBarrel == ((rawId >> DetId::kSubdetOffset) & DetId::kSubdetMask));
    }

    ALPAKA_FN_ACC ::pixelDetails::DetIdGPU getRawId(const SiPixelMappingSoAConstView &cablingMap,
                                                    uint8_t fed,
                                                    uint32_t link,
                                                    uint32_t roc) {
      using namespace ::pixelDetails;
      uint32_t index = fed * MAX_LINK * MAX_ROC + (link - 1) * MAX_ROC + roc;
      DetIdGPU detId = {cablingMap.rawId()[index], cablingMap.rocInDet()[index], cablingMap.moduleId()[index]};
      return detId;
    }

    //reference http://cmsdoxygen.web.cern.ch/cmsdoxygen/CMSSW_9_2_0/doc/html/dd/d31/FrameConversion_8cc_source.html
    //http://cmslxr.fnal.gov/source/CondFormats/SiPixelObjects/src/PixelROC.cc?v=CMSSW_9_2_0#0071
    // Convert local pixel to pixelDetails::global pixel
    ALPAKA_FN_ACC ::pixelDetails::Pixel frameConversion(
        bool bpix, int side, uint32_t layer, uint32_t rocIdInDetUnit, ::pixelDetails::Pixel local) {
      int slopeRow = 0, slopeCol = 0;
      int rowOffset = 0, colOffset = 0;

      if (bpix) {
        if (side == -1 && layer != 1) {  // -Z side: 4 non-flipped modules oriented like 'dddd', except Layer 1
          if (rocIdInDetUnit < 8) {
            slopeRow = 1;
            slopeCol = -1;
            rowOffset = 0;
            colOffset = (8 - rocIdInDetUnit) * ::pixelDetails::numColsInRoc - 1;
          } else {
            slopeRow = -1;
            slopeCol = 1;
            rowOffset = 2 * ::pixelDetails::numRowsInRoc - 1;
            colOffset = (rocIdInDetUnit - 8) * ::pixelDetails::numColsInRoc;
          }  // if roc
        } else {  // +Z side: 4 non-flipped modules oriented like 'pppp', but all 8 in layer1
          if (rocIdInDetUnit < 8) {
            slopeRow = -1;
            slopeCol = 1;
            rowOffset = 2 * ::pixelDetails::numRowsInRoc - 1;
            colOffset = rocIdInDetUnit * ::pixelDetails::numColsInRoc;
          } else {
            slopeRow = 1;
            slopeCol = -1;
            rowOffset = 0;
            colOffset = (16 - rocIdInDetUnit) * ::pixelDetails::numColsInRoc - 1;
          }
        }

      } else {             // fpix
        if (side == -1) {  // pannel 1
          if (rocIdInDetUnit < 8) {
            slopeRow = 1;
            slopeCol = -1;
            rowOffset = 0;
            colOffset = (8 - rocIdInDetUnit) * ::pixelDetails::numColsInRoc - 1;
          } else {
            slopeRow = -1;
            slopeCol = 1;
            rowOffset = 2 * ::pixelDetails::numRowsInRoc - 1;
            colOffset = (rocIdInDetUnit - 8) * ::pixelDetails::numColsInRoc;
          }
        } else {  // pannel 2
          if (rocIdInDetUnit < 8) {
            slopeRow = 1;
            slopeCol = -1;
            rowOffset = 0;
            colOffset = (8 - rocIdInDetUnit) * ::pixelDetails::numColsInRoc - 1;
          } else {
            slopeRow = -1;
            slopeCol = 1;
            rowOffset = 2 * ::pixelDetails::numRowsInRoc - 1;
            colOffset = (rocIdInDetUnit - 8) * ::pixelDetails::numColsInRoc;
          }

        }  // side
      }

      uint32_t gRow = rowOffset + slopeRow * local.row;
      uint32_t gCol = colOffset + slopeCol * local.col;
      // inside frameConversion row: gRow, column: gCol
      ::pixelDetails::Pixel global = {gRow, gCol};
      return global;
    }

    // error decoding and handling copied from EventFilter/SiPixelRawToDigi/src/ErrorChecker.cc
    template <bool debug = false>
    ALPAKA_FN_ACC uint8_t conversionError(uint8_t fedId, uint8_t status) {
      uint8_t errorType = 0;

      switch (status) {
        case 1: {
          if constexpr (debug)
            printf("Error in Fed: %i, invalid channel Id (errorType = 35\n)", fedId);
          errorType = 35;
          break;
        }
        case 2: {
          if constexpr (debug)
            printf("Error in Fed: %i, invalid ROC Id (errorType = 36)\n", fedId);
          errorType = 36;
          break;
        }
        case 3: {
          if constexpr (debug)
            printf("Error in Fed: %i, invalid dcol/pixel value (errorType = 37)\n", fedId);
          errorType = 37;
          break;
        }
        case 4: {
          if constexpr (debug)
            printf("Error in Fed: %i, dcol/pixel read out of order (errorType = 38)\n", fedId);
          errorType = 38;
          break;
        }
        default:
          if constexpr (debug)
            printf("Cabling check returned unexpected result, status = %i\n", status);
      };

      return errorType;
    }

    ALPAKA_FN_ACC bool rocRowColIsValid(uint32_t rocRow, uint32_t rocCol) {
      /// row and column in ROC representation
      return ((rocRow < ::pixelDetails::numRowsInRoc) & (rocCol < ::pixelDetails::numColsInRoc));
    }

    ALPAKA_FN_ACC bool dcolIsValid(uint32_t dcol, uint32_t pxid) { return ((dcol < 26) & (2 <= pxid) & (pxid < 162)); }

    // error decoding and handling copied from EventFilter/SiPixelRawToDigi/src/ErrorChecker.cc
    template <bool debug = false>
    ALPAKA_FN_ACC uint8_t
    checkROC(uint32_t errorWord, uint8_t fedId, uint32_t link, const SiPixelMappingSoAConstView &cablingMap) {
      uint8_t errorType = (errorWord >> ::pixelDetails::ROC_shift) & ::pixelDetails::ERROR_mask;
      if (errorType < 25)
        return 0;
      bool errorFound = false;

      switch (errorType) {
        case 25: {
          errorFound = true;
          uint32_t index =
              fedId * ::pixelDetails::MAX_LINK * ::pixelDetails::MAX_ROC + (link - 1) * ::pixelDetails::MAX_ROC + 1;
          if (index > 1 && index <= cablingMap.size()) {
            if (!(link == cablingMap.link()[index] && 1 == cablingMap.roc()[index]))
              errorFound = false;
          }
          if constexpr (debug)
            if (errorFound)
              printf("Invalid ROC = 25 found (errorType = 25)\n");
          break;
        }
        case 26: {
          if constexpr (debug)
            printf("Gap word found (errorType = 26)\n");
          break;
        }
        case 27: {
          if constexpr (debug)
            printf("Dummy word found (errorType = 27)\n");
          break;
        }
        case 28: {
          if constexpr (debug)
            printf("Error fifo nearly full (errorType = 28)\n");
          errorFound = true;
          break;
        }
        case 29: {
          if constexpr (debug)
            printf("Timeout on a channel (errorType = 29)\n");
          if (!((errorWord >> sipixelconstants::OMIT_ERR_shift) & sipixelconstants::OMIT_ERR_mask)) {
            if constexpr (debug)
              printf("...2nd errorType=29 error, skip\n");
            break;
          }
          errorFound = true;
          break;
        }
        case 30: {
          if constexpr (debug)
            printf("TBM error trailer (errorType = 30)\n");
          int stateMatch_bits = 4;
          int stateMatch_shift = 8;
          uint32_t stateMatch_mask = ~(~uint32_t(0) << stateMatch_bits);
          int stateMatch = (errorWord >> stateMatch_shift) & stateMatch_mask;
          if (stateMatch != 1 && stateMatch != 8) {
            if constexpr (debug)
              printf("FED error 30 with unexpected State Bits (errorType = 30)\n");
            break;
          }
          if (stateMatch == 1)
            errorType = 40;  // 1=Overflow -> 40, 8=number of ROCs -> 30
          errorFound = true;
          break;
        }
        case 31: {
          if constexpr (debug)
            printf("Event number error (errorType = 31)\n");
          errorFound = true;
          break;
        }
        default:
          errorFound = false;
      };

      return errorFound ? errorType : 0;
    }

    // error decoding and handling copied from EventFilter/SiPixelRawToDigi/src/ErrorChecker.cc
    template <bool debug = false>
    ALPAKA_FN_ACC uint32_t
    getErrRawID(uint8_t fedId, uint32_t errWord, uint32_t errorType, const SiPixelMappingSoAConstView &cablingMap) {
      uint32_t rID = 0xffffffff;

      switch (errorType) {
        case 25:
        case 29:
        case 30:
        case 31:
        case 36:
        case 40: {
          uint32_t roc = 1;
          uint32_t link = (errWord >> ::pixelDetails::LINK_shift) & ::pixelDetails::LINK_mask;
          uint32_t rID_temp = getRawId(cablingMap, fedId, link, roc).rawId;
          if (rID_temp != ::pixelClustering::invalidModuleId)
            rID = rID_temp;
          break;
        }
        case 37:
        case 38: {
          uint32_t roc = (errWord >> ::pixelDetails::ROC_shift) & ::pixelDetails::ROC_mask;
          uint32_t link = (errWord >> ::pixelDetails::LINK_shift) & ::pixelDetails::LINK_mask;
          uint32_t rID_temp = getRawId(cablingMap, fedId, link, roc).rawId;
          if (rID_temp != ::pixelClustering::invalidModuleId)
            rID = rID_temp;
          break;
        }
        default:
          break;
      };

      return rID;
    }

    // Kernel to perform Raw to Digi conversion
    template <bool debug = false>
    struct RawToDigi_kernel {
      ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                    const SiPixelMappingSoAConstView &cablingMap,
                                    const unsigned char *modToUnp,
                                    const uint32_t wordCounter,
                                    const uint32_t *word,
                                    const uint8_t *fedIds,
                                    SiPixelDigisSoAView digisView,
                                    SiPixelDigiErrorsSoAView err,
                                    bool useQualityInfo,
                                    bool includeErrors) const {
        // FIXME there is no guarantee that this is initialised to 0 before any of the atomicInc happens
        if (cms::alpakatools::once_per_grid(acc))
          err.size() = 0;

        for (auto gIndex : cms::alpakatools::uniform_elements(acc, wordCounter)) {
          auto dvgi = digisView[gIndex];
          dvgi.xx() = 0;
          dvgi.yy() = 0;
          dvgi.adc() = 0;

          // initialise the errors
          err[gIndex].pixelErrors() = SiPixelErrorCompact{0, 0, 0, 0};

          uint8_t fedId = fedIds[gIndex / 2];  // +1200;

          // initialize (too many coninue below)
          dvgi.pdigi() = 0;
          dvgi.rawIdArr() = 0;
          dvgi.moduleId() = ::pixelClustering::invalidModuleId;

          uint32_t ww = word[gIndex];  // Array containing 32 bit raw data
          if (ww == 0) {
            // 0 is an indicator of a noise/dead channel, skip these pixels during clusterization
            continue;
          }

          uint32_t link = sipixelconstants::getLink(ww);  // Extract link
          uint32_t roc = sipixelconstants::getROC(ww);    // Extract ROC in link

          uint8_t errorType = checkROC<debug>(ww, fedId, link, cablingMap);
          bool skipROC = (roc < ::pixelDetails::maxROCIndex) ? false : (errorType != 0);
          if (includeErrors and skipROC) {
            uint32_t rawId = getErrRawID<debug>(fedId, ww, errorType, cablingMap);
            if (rawId != 0xffffffff)  // Store errors only for valid DetIds
            {
              err[gIndex].pixelErrors() = SiPixelErrorCompact{rawId, ww, errorType, fedId};
              alpaka::atomicInc(acc, &err.size(), 0xffffffff, alpaka::hierarchy::Blocks{});
            }
            continue;
          }

          // Check for spurious channels
          if (roc > ::pixelDetails::MAX_ROC or link > ::pixelDetails::MAX_LINK) {
            uint32_t rawId = getRawId(cablingMap, fedId, link, 1).rawId;
            if constexpr (debug) {
              printf("spurious roc %d found on link %d, detector %d (index %d)\n", roc, link, rawId, gIndex);
            }
            if (roc > ::pixelDetails::MAX_ROC and roc < 25) {
              uint8_t error = conversionError<debug>(fedId, 2);
              err[gIndex].pixelErrors() = SiPixelErrorCompact{rawId, ww, error, fedId};
              alpaka::atomicInc(acc, &err.size(), 0xffffffff, alpaka::hierarchy::Blocks{});
            }
            continue;
          }

          uint32_t index =
              fedId * ::pixelDetails::MAX_LINK * ::pixelDetails::MAX_ROC + (link - 1) * ::pixelDetails::MAX_ROC + roc;
          if (useQualityInfo) {
            skipROC = cablingMap.badRocs()[index];
            if (skipROC)
              continue;
          }
          skipROC = modToUnp[index];
          if (skipROC)
            continue;

          ::pixelDetails::DetIdGPU detId = getRawId(cablingMap, fedId, link, roc);
          uint32_t rawId = detId.rawId;
          uint32_t layer = 0;
          int side = 0, panel = 0, module = 0;
          bool barrel = isBarrel(rawId);

          if (barrel) {
            layer = (rawId >> ::pixelDetails::layerStartBit) & ::pixelDetails::layerMask;
            module = (rawId >> ::pixelDetails::moduleStartBit) & ::pixelDetails::moduleMask;
            side = (module < 5) ? -1 : 1;
          } else {
            // endcap ids
            layer = 0;
            panel = (rawId >> ::pixelDetails::panelStartBit) & ::pixelDetails::panelMask;
            side = (panel == 1) ? -1 : 1;
          }

          ::pixelDetails::Pixel localPix;
          if (layer == 1) {
            // Special case of barrel layer 1
            uint32_t col = sipixelconstants::getCol(ww);
            uint32_t row = sipixelconstants::getRow(ww);
            localPix.row = row;
            localPix.col = col;
            if (includeErrors and not rocRowColIsValid(row, col)) {
              uint8_t error = conversionError<debug>(fedId, 3);
              err[gIndex].pixelErrors() = SiPixelErrorCompact{rawId, ww, error, fedId};
              alpaka::atomicInc(acc, &err.size(), 0xffffffff, alpaka::hierarchy::Blocks{});
              if constexpr (debug)
                printf("BPIX1 Error status: %i\n", error);
              continue;
            }
          } else {
            // Other layers with double columns
            uint32_t dcol = sipixelconstants::getDCol(ww);
            uint32_t pxid = sipixelconstants::getPxId(ww);
            uint32_t row = ::pixelDetails::numRowsInRoc - pxid / 2;
            uint32_t col = dcol * 2 + pxid % 2;
            localPix.row = row;
            localPix.col = col;
            if (includeErrors and not dcolIsValid(dcol, pxid)) {
              uint8_t error = conversionError<debug>(fedId, 3);
              err[gIndex].pixelErrors() = SiPixelErrorCompact{rawId, ww, error, fedId};
              alpaka::atomicInc(acc, &err.size(), 0xffffffff, alpaka::hierarchy::Blocks{});
              if constexpr (debug)
                printf("Error status: %i %d %d %d %d\n", error, dcol, pxid, fedId, roc);
              continue;
            }
          }

          ::pixelDetails::Pixel globalPix = frameConversion(barrel, side, layer, detId.rocInDet, localPix);
          dvgi.xx() = globalPix.row;  // origin shifting by 1 0-159
          dvgi.yy() = globalPix.col;  // origin shifting by 1 0-415
          dvgi.adc() = sipixelconstants::getADC(ww);
          dvgi.pdigi() = ::pixelDetails::pack(globalPix.row, globalPix.col, dvgi.adc());
          dvgi.moduleId() = detId.moduleId;
          dvgi.rawIdArr() = rawId;
        }  // end of stride on grid

      }  // end of Raw to Digi kernel operator()
    };  // end of Raw to Digi struct

    template <typename TrackerTraits>
    struct FillHitsModuleStart {
      ALPAKA_FN_ACC void operator()(Acc1D const &acc, SiPixelClustersSoAView clus_view) const {
        // This kernel must run with a single block
        [[maybe_unused]] const uint32_t blockIdxLocal(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
        ALPAKA_ASSERT_ACC(0 == blockIdxLocal);
        [[maybe_unused]] const uint32_t gridDimension(alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
        ALPAKA_ASSERT_ACC(1 == gridDimension);

        // For the prefix scan algorithm
        constexpr int warpSize = cms::alpakatools::warpSize;
        constexpr int blockSize = warpSize * warpSize;

        // For Phase1 there are 1856 pixel modules
        // For Phase2 there are up to 4000 pixel modules
        constexpr uint16_t numberOfModules = TrackerTraits::numberOfModules;
        constexpr uint16_t prefixScanUpperLimit = ((numberOfModules / blockSize) + 1) * blockSize;
        ALPAKA_ASSERT_ACC(numberOfModules < prefixScanUpperLimit);

        // Limit to maxHitsInModule;
        constexpr uint32_t maxHitsInModule = TrackerTraits::maxHitsInModule;
        for (uint32_t i : cms::alpakatools::independent_group_elements(acc, numberOfModules)) {
          clus_view[i + 1].clusModuleStart() = std::min(maxHitsInModule, clus_view[i].clusInModule());
        }

        // Use N single-block prefix scan, then update all blocks after the first one.
        auto &ws = alpaka::declareSharedVar<uint32_t[warpSize], __COUNTER__>(acc);
        uint32_t *clusModuleStart = clus_view.clusModuleStart() + 1;
        uint16_t leftModules = numberOfModules;
        while (leftModules > blockSize) {
          cms::alpakatools::blockPrefixScan(acc, clusModuleStart, clusModuleStart, blockSize, ws);
          clusModuleStart += blockSize;
          leftModules -= blockSize;
        }
        cms::alpakatools::blockPrefixScan(acc, clusModuleStart, clusModuleStart, leftModules, ws);

        // The first blockSize modules are properly accounted by the blockPrefixScan.
        // The additional modules need to be corrected adding the cuulative value from the last module of the previous block.
        for (uint16_t doneModules = blockSize; doneModules < numberOfModules; doneModules += blockSize) {
          uint16_t first = doneModules + 1;
          uint16_t last = std::min<uint16_t>(doneModules + blockSize, numberOfModules);
          for (uint16_t i : cms::alpakatools::independent_group_elements(acc, first, last + 1)) {
            clus_view[i].clusModuleStart() += clus_view[doneModules].clusModuleStart();
          }
          alpaka::syncBlockThreads(acc);
        }

#ifdef GPU_DEBUG
        ALPAKA_ASSERT_ACC(0 == clus_view[1].moduleStart());
        auto c0 = std::min(maxHitsInModule, clus_view[2].clusModuleStart());
        ALPAKA_ASSERT_ACC(c0 == clus_view[2].moduleStart());
        ALPAKA_ASSERT_ACC(clus_view[1024].moduleStart() >= clus_view[1023].moduleStart());
        ALPAKA_ASSERT_ACC(clus_view[1025].moduleStart() >= clus_view[1024].moduleStart());
        ALPAKA_ASSERT_ACC(clus_view[numberOfModules].moduleStart() >= clus_view[1025].moduleStart());

        for (uint32_t i : cms::alpakatools::independent_group_elements(acc, numberOfModules + 1)) {
          if (0 != i)
            ALPAKA_ASSERT_ACC(clus_view[i].moduleStart() >= clus_view[i - 1].moduleStart());
          // Check BPX2 (1), FP1 (4)
          constexpr auto bpix2 = TrackerTraits::layerStart[1];
          constexpr auto fpix1 = TrackerTraits::layerStart[4];
          if (i == bpix2 || i == fpix1)
            printf("moduleStart %d %d\n", i, clus_view[i].moduleStart());
        }
#endif

      }  // end of FillHitsModuleStart kernel operator()
    };  // end of FillHitsModuleStart struct

    // Interface to outside
    template <typename TrackerTraits>
    void SiPixelRawToClusterKernel<TrackerTraits>::makePhase1ClustersAsync(
        Queue &queue,
        const SiPixelClusterThresholds clusterThresholds,
        const SiPixelMappingSoAConstView &cablingMap,
        const unsigned char *modToUnp,
        const SiPixelGainCalibrationForHLTSoAConstView &gains,
        const WordFedAppender &wordFed,
        const uint32_t wordCounter,
        const uint32_t fedCounter,
        bool useQualityInfo,
        bool includeErrors,
        bool debug) {
      nDigis = wordCounter;

#ifdef GPU_DEBUG
      std::cout << "decoding " << wordCounter << " digis." << std::endl;
#endif
      constexpr int numberOfModules = TrackerTraits::numberOfModules;
      digis_d = SiPixelDigisSoACollection(wordCounter, queue);
      if (includeErrors) {
        digiErrors_d = SiPixelDigiErrorsSoACollection(wordCounter, queue);
      }
      clusters_d = SiPixelClustersSoACollection(numberOfModules, queue);
      // protect in case of empty event....
      if (wordCounter) {
        const int threadsPerBlockOrElementsPerThread =
            cms::alpakatools::requires_single_thread_per_block_v<Acc1D> ? 32 : 512;
        // fill it all
        const uint32_t blocks = cms::alpakatools::divide_up_by(wordCounter, threadsPerBlockOrElementsPerThread);
        const auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlockOrElementsPerThread);
        assert(0 == wordCounter % 2);
        // wordCounter is the total no of words in each event to be trasfered on device
        auto word_d = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, wordCounter);
        // NB: IMPORTANT: fedId_d: In legacy, wordCounter elements are allocated.
        // However, only the first half of elements end up eventually used:
        // hence, here, only wordCounter/2 elements are allocated.
        auto fedId_d = cms::alpakatools::make_device_buffer<uint8_t[]>(queue, wordCounter / 2);
        alpaka::memcpy(queue, word_d, wordFed.word(), wordCounter);
        alpaka::memcpy(queue, fedId_d, wordFed.fedId(), wordCounter / 2);
        // Launch rawToDigi kernel
        if (debug) {
          alpaka::exec<Acc1D>(queue,
                              workDiv,
                              RawToDigi_kernel<true>{},
                              cablingMap,
                              modToUnp,
                              wordCounter,
                              word_d.data(),
                              fedId_d.data(),
                              digis_d->view(),
                              digiErrors_d->view(),
                              useQualityInfo,
                              includeErrors);
        } else {
          alpaka::exec<Acc1D>(queue,
                              workDiv,
                              RawToDigi_kernel<false>{},
                              cablingMap,
                              modToUnp,
                              wordCounter,
                              word_d.data(),
                              fedId_d.data(),
                              digis_d->view(),
                              digiErrors_d->view(),
                              useQualityInfo,
                              includeErrors);
        }

#ifdef GPU_DEBUG
        alpaka::wait(queue);
        std::cout << "RawToDigi_kernel was run smoothly!" << std::endl;
#endif
      }
      // End of Raw2Digi and passing data for clustering

      {
        // clusterizer
        using namespace pixelClustering;
        // calibrations
        using namespace calibPixel;
        const int threadsPerBlockOrElementsPerThread = []() {
          if constexpr (std::is_same_v<Device, alpaka_common::DevHost>) {
            // NB: MPORTANT: This could be tuned to benefit from innermost loop.
            return 32;
          } else {
            return 256;
          }
        }();
        const auto blocks = cms::alpakatools::divide_up_by(std::max<int>(wordCounter, numberOfModules),
                                                           threadsPerBlockOrElementsPerThread);
        const auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlockOrElementsPerThread);

        if (debug) {
          alpaka::exec<Acc1D>(queue,
                              workDiv,
                              CalibDigis<true>{},
                              clusterThresholds,
                              digis_d->view(),
                              clusters_d->view(),
                              gains,
                              wordCounter);
        } else {
          alpaka::exec<Acc1D>(queue,
                              workDiv,
                              CalibDigis<false>{},
                              clusterThresholds,
                              digis_d->view(),
                              clusters_d->view(),
                              gains,
                              wordCounter);
        }
#ifdef GPU_DEBUG
        alpaka::wait(queue);
        std::cout << "CountModules kernel launch with " << blocks << " blocks of " << threadsPerBlockOrElementsPerThread
                  << " threadsPerBlockOrElementsPerThread\n";
#endif

        alpaka::exec<Acc1D>(
            queue, workDiv, CountModules<TrackerTraits>{}, digis_d->view(), clusters_d->view(), wordCounter);

        auto moduleStartFirstElement = cms::alpakatools::make_device_view(queue, clusters_d->view().moduleStart(), 1u);
        alpaka::memcpy(queue, nModules_Clusters_h, moduleStartFirstElement);

        const auto elementsPerBlockFindClus = FindClus<TrackerTraits>::maxElementsPerBlock;
        const auto workDivMaxNumModules =
            cms::alpakatools::make_workdiv<Acc1D>(numberOfModules, elementsPerBlockFindClus);
#ifdef GPU_DEBUG
        std::cout << " FindClus kernel launch with " << numberOfModules << " blocks of " << elementsPerBlockFindClus
                  << " threadsPerBlockOrElementsPerThread\n";
#endif
        alpaka::exec<Acc1D>(
            queue, workDivMaxNumModules, FindClus<TrackerTraits>{}, digis_d->view(), clusters_d->view(), wordCounter);
#ifdef GPU_DEBUG
        alpaka::wait(queue);
#endif

        constexpr auto threadsPerBlockChargeCut = 256;
        const auto workDivChargeCut = cms::alpakatools::make_workdiv<Acc1D>(numberOfModules, threadsPerBlockChargeCut);
        // apply charge cut
        alpaka::exec<Acc1D>(queue,
                            workDivChargeCut,
                            ::pixelClustering::ClusterChargeCut<TrackerTraits>{},
                            digis_d->view(),
                            clusters_d->view(),
                            clusterThresholds,
                            wordCounter);
        // count the module start indices already here (instead of
        // rechits) so that the number of clusters/hits can be made
        // available in the rechit producer without additional points of
        // synchronization/ExternalWork

        // MUST be ONE block
        const auto workDivOneBlock = cms::alpakatools::make_workdiv<Acc1D>(1u, 1024u);
        alpaka::exec<Acc1D>(queue, workDivOneBlock, FillHitsModuleStart<TrackerTraits>{}, clusters_d->view());

        // last element holds the number of all clusters
        const auto clusModuleStartLastElement =
            cms::alpakatools::make_device_view(queue, clusters_d->const_view().clusModuleStart() + numberOfModules, 1u);
        constexpr int startBPIX2 = TrackerTraits::layerStart[1];

        // element startBPIX2 hold the number of clusters until BPIX2
        const auto bpix2ClusterStart =
            cms::alpakatools::make_device_view(queue, clusters_d->const_view().clusModuleStart() + startBPIX2, 1u);
        auto nModules_Clusters_h_1 = cms::alpakatools::make_host_view(nModules_Clusters_h.data() + 1, 1u);
        alpaka::memcpy(queue, nModules_Clusters_h_1, clusModuleStartLastElement);

        auto nModules_Clusters_h_2 = cms::alpakatools::make_host_view(nModules_Clusters_h.data() + 2, 1u);
        alpaka::memcpy(queue, nModules_Clusters_h_2, bpix2ClusterStart);

#ifdef GPU_DEBUG
        alpaka::wait(queue);
        std::cout << "SiPixelClusterizerAlpaka results:" << std::endl
                  << " > no. of digis: " << nDigis << std::endl
                  << " > no. of active modules: " << nModules_Clusters_h[0] << std::endl
                  << " > no. of clusters: " << nModules_Clusters_h[1] << std::endl
                  << " > bpix2 offset: " << nModules_Clusters_h[2] << std::endl;
#endif

      }  // end clusterizer scope
    }

    template <typename TrackerTraits>
    void SiPixelRawToClusterKernel<TrackerTraits>::makePhase2ClustersAsync(
        Queue &queue,
        const SiPixelClusterThresholds clusterThresholds,
        SiPixelDigisSoAView &digis_view,
        const uint32_t numDigis) {
      using namespace pixelClustering;
      using pixelTopology::Phase2;
      nDigis = numDigis;
      constexpr int numberOfModules = pixelTopology::Phase2::numberOfModules;
      clusters_d = SiPixelClustersSoACollection(numberOfModules, queue);
      const auto threadsPerBlockOrElementsPerThread = 512;
      const auto blocks =
          cms::alpakatools::divide_up_by(std::max<int>(numDigis, numberOfModules), threadsPerBlockOrElementsPerThread);
      const auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlockOrElementsPerThread);

      alpaka::exec<Acc1D>(
          queue, workDiv, calibPixel::CalibDigisPhase2{}, clusterThresholds, digis_view, clusters_d->view(), numDigis);

#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "CountModules kernel launch with " << blocks << " blocks of " << threadsPerBlockOrElementsPerThread
                << " threadsPerBlockOrElementsPerThread\n";
#endif
      alpaka::exec<Acc1D>(
          queue, workDiv, CountModules<pixelTopology::Phase2>{}, digis_view, clusters_d->view(), numDigis);

      auto moduleStartFirstElement = cms::alpakatools::make_device_view(queue, clusters_d->view().moduleStart(), 1u);
      alpaka::memcpy(queue, nModules_Clusters_h, moduleStartFirstElement);

      const auto elementsPerBlockFindClus = FindClus<TrackerTraits>::maxElementsPerBlock;
      const auto workDivMaxNumModules =
          cms::alpakatools::make_workdiv<Acc1D>(numberOfModules, elementsPerBlockFindClus);
#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "FindClus kernel launch with " << numberOfModules << " blocks of " << elementsPerBlockFindClus
                << " threadsPerBlockOrElementsPerThread\n";
#endif
      alpaka::exec<Acc1D>(
          queue, workDivMaxNumModules, FindClus<TrackerTraits>{}, digis_view, clusters_d->view(), numDigis);
#ifdef GPU_DEBUG
      alpaka::wait(queue);
#endif

      // apply charge cut
      alpaka::exec<Acc1D>(queue,
                          workDivMaxNumModules,
                          ::pixelClustering::ClusterChargeCut<TrackerTraits>{},
                          digis_view,
                          clusters_d->view(),
                          clusterThresholds,
                          numDigis);

      // count the module start indices already here (instead of
      // rechits) so that the number of clusters/hits can be made
      // available in the rechit producer without additional points of
      // synchronization/ExternalWork

      // MUST be ONE block
      const auto workDivOneBlock = cms::alpakatools::make_workdiv<Acc1D>(1u, 1024u);
      alpaka::exec<Acc1D>(queue, workDivOneBlock, FillHitsModuleStart<TrackerTraits>{}, clusters_d->view());

      // last element holds the number of all clusters
      const auto clusModuleStartLastElement =
          cms::alpakatools::make_device_view(queue, clusters_d->const_view().clusModuleStart() + numberOfModules, 1u);
      constexpr int startBPIX2 = pixelTopology::Phase2::layerStart[1];
      // element startBPIX2 hold the number of clusters until BPIX2
      const auto bpix2ClusterStart =
          cms::alpakatools::make_device_view(queue, clusters_d->const_view().clusModuleStart() + startBPIX2, 1u);
      auto nModules_Clusters_h_1 = cms::alpakatools::make_host_view(nModules_Clusters_h.data() + 1, 1u);
      alpaka::memcpy(queue, nModules_Clusters_h_1, clusModuleStartLastElement);

      auto nModules_Clusters_h_2 = cms::alpakatools::make_host_view(nModules_Clusters_h.data() + 2, 1u);
      alpaka::memcpy(queue, nModules_Clusters_h_2, bpix2ClusterStart);

#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "SiPixelPhase2DigiToCluster: results \n"
                << " > no. of digis: " << numDigis << std::endl
                << " > no. of active modules: " << nModules_Clusters_h[0] << std::endl
                << " > no. of clusters: " << nModules_Clusters_h[1] << std::endl
                << " > bpix2 offset: " << nModules_Clusters_h[2] << std::endl;
#endif
    }  //

    template class SiPixelRawToClusterKernel<pixelTopology::Phase1>;
    template class SiPixelRawToClusterKernel<pixelTopology::Phase2>;
    template class SiPixelRawToClusterKernel<pixelTopology::HIonPhase1>;

  }  // namespace pixelDetails

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
