// C++ includes
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>

// CMSSW includes
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/SimpleVector.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationForHLTLayout.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelMappingLayout.h"
#include "DataFormats/SiPixelDigi/interface/SiPixelDigiConstants.h"

// local includes
#include "CalibPixel.h"
#include "ClusterChargeCut.h"
#include "PixelClustering.h"
#include "SiPixelRawToClusterKernel.h"

// #define GPU_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace pixelDetails {

    ////////////////////

    ALPAKA_FN_ACC uint32_t getLink(uint32_t ww) {
      return ((ww >> ::sipixelconstants::LINK_shift) & ::sipixelconstants::LINK_mask);
    }

    ALPAKA_FN_ACC uint32_t getRoc(uint32_t ww) {
      return ((ww >> ::sipixelconstants::ROC_shift) & ::sipixelconstants::ROC_mask);
    }

    ALPAKA_FN_ACC uint32_t getADC(uint32_t ww) {
      return ((ww >> ::sipixelconstants::ADC_shift) & ::sipixelconstants::ADC_mask);
    }

    ALPAKA_FN_ACC bool isBarrel(uint32_t rawId) { return (1 == ((rawId >> 25) & 0x7)); }

    ALPAKA_FN_ACC ::pixelDetails::DetIdGPU getRawId(const SiPixelMappingSoAConstView &cablingMap,
                                                    uint8_t fed,
                                                    uint32_t link,
                                                    uint32_t roc) {
      using namespace ::pixelDetails;
      uint32_t index = fed * MAX_LINK * MAX_ROC + (link - 1) * MAX_ROC + roc;
      ::pixelDetails::DetIdGPU detId = {
          cablingMap.rawId()[index], cablingMap.rocInDet()[index], cablingMap.moduleId()[index]};
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
          }       // if roc
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
      ::pixelDetails::Pixel global = {gRow, gCol};
      return global;
    }

    ALPAKA_FN_ACC uint8_t conversionError(uint8_t fedId, uint8_t status, bool debug = false) {
      uint8_t errorType = 0;

      switch (status) {
        case 1: {
          if (debug)
            printf("Error in Fed: %i, invalid channel Id (errorType = 35\n)", fedId);
          errorType = 35;
          break;
        }
        case 2: {
          if (debug)
            printf("Error in Fed: %i, invalid ROC Id (errorType = 36)\n", fedId);
          errorType = 36;
          break;
        }
        case 3: {
          if (debug)
            printf("Error in Fed: %i, invalid dcol/pixel value (errorType = 37)\n", fedId);
          errorType = 37;
          break;
        }
        case 4: {
          if (debug)
            printf("Error in Fed: %i, dcol/pixel read out of order (errorType = 38)\n", fedId);
          errorType = 38;
          break;
        }
        default:
          if (debug)
            printf("Cabling check returned unexpected result, status = %i\n", status);
      };

      return errorType;
    }

    ALPAKA_FN_ACC bool rocRowColIsValid(uint32_t rocRow, uint32_t rocCol) {
      uint32_t numRowsInRoc = 80;
      uint32_t numColsInRoc = 52;

      /// row and collumn in ROC representation
      return ((rocRow < numRowsInRoc) & (rocCol < numColsInRoc));
    }

    ALPAKA_FN_ACC bool dcolIsValid(uint32_t dcol, uint32_t pxid) { return ((dcol < 26) & (2 <= pxid) & (pxid < 162)); }

    ALPAKA_FN_ACC uint8_t checkROC(uint32_t errorWord,
                                   uint8_t fedId,
                                   uint32_t link,
                                   const SiPixelMappingSoAConstView &cablingMap,
                                   bool debug = false) {
      uint8_t errorType = (errorWord >> ::pixelDetails::ROC_shift) & ::pixelDetails::ERROR_mask;
      if (errorType < 25)
        return 0;
      bool errorFound = false;

      switch (errorType) {
        case (25): {
          errorFound = true;
          uint32_t index =
              fedId * ::pixelDetails::MAX_LINK * ::pixelDetails::MAX_ROC + (link - 1) * ::pixelDetails::MAX_ROC + 1;
          if (index > 1 && index <= cablingMap.size()) {
            if (!(link == cablingMap.link()[index] && 1 == cablingMap.roc()[index]))
              errorFound = false;
          }
          if (debug and errorFound)
            printf("Invalid ROC = 25 found (errorType = 25)\n");
          break;
        }
        case (26): {
          if (debug)
            printf("Gap word found (errorType = 26)\n");
          errorFound = true;
          break;
        }
        case (27): {
          if (debug)
            printf("Dummy word found (errorType = 27)\n");
          errorFound = true;
          break;
        }
        case (28): {
          if (debug)
            printf("Error fifo nearly full (errorType = 28)\n");
          errorFound = true;
          break;
        }
        case (29): {
          if (debug)
            printf("Timeout on a channel (errorType = 29)\n");
          if ((errorWord >> ::pixelDetails::OMIT_ERR_shift) & ::pixelDetails::OMIT_ERR_mask) {
            if (debug)
              printf("...first errorType=29 error, this gets masked out\n");
          }
          errorFound = true;
          break;
        }
        case (30): {
          if (debug)
            printf("TBM error trailer (errorType = 30)\n");
          int StateMatch_bits = 4;
          int StateMatch_shift = 8;
          uint32_t StateMatch_mask = ~(~uint32_t(0) << StateMatch_bits);
          int StateMatch = (errorWord >> StateMatch_shift) & StateMatch_mask;
          if (StateMatch != 1 && StateMatch != 8) {
            if (debug)
              printf("FED error 30 with unexpected State Bits (errorType = 30)\n");
          }
          if (StateMatch == 1)
            errorType = 40;  // 1=Overflow -> 40, 8=number of ROCs -> 30
          errorFound = true;
          break;
        }
        case (31): {
          if (debug)
            printf("Event number error (errorType = 31)\n");
          errorFound = true;
          break;
        }
        default:
          errorFound = false;
      };

      return errorFound ? errorType : 0;
    }

    ALPAKA_FN_ACC uint32_t getErrRawID(uint8_t fedId,
                                       uint32_t errWord,
                                       uint32_t errorType,
                                       const SiPixelMappingSoAConstView &cablingMap,
                                       bool debug = false) {
      uint32_t rID = 0xffffffff;

      switch (errorType) {
        case 25:
        case 30:
        case 31:
        case 36:
        case 40: {
          uint32_t roc = 1;
          uint32_t link = (errWord >> ::pixelDetails::LINK_shift) & ::pixelDetails::LINK_mask;
          uint32_t rID_temp = getRawId(cablingMap, fedId, link, roc).RawId;
          if (rID_temp != 9999)
            rID = rID_temp;
          break;
        }
        case 29: {
          int chanNmbr = 0;
          const int DB0_shift = 0;
          const int DB1_shift = DB0_shift + 1;
          const int DB2_shift = DB1_shift + 1;
          const int DB3_shift = DB2_shift + 1;
          const int DB4_shift = DB3_shift + 1;
          const uint32_t DataBit_mask = ~(~uint32_t(0) << 1);

          int CH1 = (errWord >> DB0_shift) & DataBit_mask;
          int CH2 = (errWord >> DB1_shift) & DataBit_mask;
          int CH3 = (errWord >> DB2_shift) & DataBit_mask;
          int CH4 = (errWord >> DB3_shift) & DataBit_mask;
          int CH5 = (errWord >> DB4_shift) & DataBit_mask;
          int BLOCK_bits = 3;
          int BLOCK_shift = 8;
          uint32_t BLOCK_mask = ~(~uint32_t(0) << BLOCK_bits);
          int BLOCK = (errWord >> BLOCK_shift) & BLOCK_mask;
          int localCH = 1 * CH1 + 2 * CH2 + 3 * CH3 + 4 * CH4 + 5 * CH5;
          if (BLOCK % 2 == 0)
            chanNmbr = (BLOCK / 2) * 9 + localCH;
          else
            chanNmbr = ((BLOCK - 1) / 2) * 9 + 4 + localCH;
          if ((chanNmbr < 1) || (chanNmbr > 36))
            break;  // signifies unexpected result

          uint32_t roc = 1;
          uint32_t link = chanNmbr;
          uint32_t rID_temp = getRawId(cablingMap, fedId, link, roc).RawId;
          if (rID_temp != 9999)
            rID = rID_temp;
          break;
        }
        case 37:
        case 38: {
          uint32_t roc = (errWord >> ::pixelDetails::ROC_shift) & ::pixelDetails::ROC_mask;
          uint32_t link = (errWord >> ::pixelDetails::LINK_shift) & ::pixelDetails::LINK_mask;
          uint32_t rID_temp = getRawId(cablingMap, fedId, link, roc).RawId;
          if (rID_temp != 9999)
            rID = rID_temp;
          break;
        }
        default:
          break;
      };

      return rID;
    }

    // Kernel to perform Raw to Digi conversion
    struct RawToDigi_kernel {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                    const SiPixelMappingSoAConstView &cablingMap,
                                    const unsigned char *modToUnp,
                                    const uint32_t wordCounter,
                                    const uint32_t *word,
                                    const uint8_t *fedIds,
                                    SiPixelDigisSoAView digisView,
                                    SiPixelDigiErrorsSoAView err,
                                    bool useQualityInfo,
                                    bool includeErrors,
                                    bool debug) const {
        cms::alpakatools::for_each_element_in_grid_strided(acc, wordCounter, [&](uint32_t iloop) {
          auto gIndex = iloop;
          auto dvgi = digisView[gIndex];
          dvgi.xx() = 0;
          dvgi.yy() = 0;
          dvgi.adc() = 0;
          bool skipROC = false;

          if (gIndex == 0)
            err[gIndex].size() = 0;

          err[gIndex].pixelErrors() = SiPixelErrorCompact{0, 0, 0, 0};

          uint8_t fedId = fedIds[gIndex / 2];  // +1200;

          // initialize (too many coninue below)
          dvgi.pdigi() = 0;
          dvgi.rawIdArr() = 0;
          constexpr uint16_t invalidModuleId = std::numeric_limits<uint16_t>::max() - 1;
          dvgi.moduleId() = invalidModuleId;

          uint32_t ww = word[gIndex];  // Array containing 32 bit raw data
          if (ww == 0) {
            // 0 is an indicator of a noise/dead channel, skip these pixels during clusterization
            return;
          }

          uint32_t link = getLink(ww);  // Extract link
          uint32_t roc = getRoc(ww);    // Extract Roc in link
          ::pixelDetails::DetIdGPU detId = getRawId(cablingMap, fedId, link, roc);

          uint8_t errorType = checkROC(ww, fedId, link, cablingMap, debug);
          skipROC = (roc < ::pixelDetails::maxROCIndex) ? false : (errorType != 0);
          if (includeErrors and skipROC) {
            uint32_t rID = getErrRawID(fedId, ww, errorType, cablingMap, debug);
            err[gIndex].pixelErrors() = SiPixelErrorCompact{rID, ww, errorType, fedId};
            alpaka::atomicInc(acc, &err.size(), 0xffffffff, alpaka::hierarchy::Threads{});
            return;
          }

          uint32_t rawId = detId.RawId;
          uint32_t rocIdInDetUnit = detId.rocInDet;
          bool barrel = isBarrel(rawId);

          uint32_t index =
              fedId * ::pixelDetails::MAX_LINK * ::pixelDetails::MAX_ROC + (link - 1) * ::pixelDetails::MAX_ROC + roc;
          if (useQualityInfo) {
            skipROC = cablingMap.badRocs()[index];
            if (skipROC)
              return;
          }
          skipROC = modToUnp[index];
          if (skipROC)
            return;

          uint32_t layer = 0;                   //, ladder =0;
          int side = 0, panel = 0, module = 0;  //disk = 0, blade = 0

          if (barrel) {
            layer = (rawId >> ::pixelDetails::layerStartBit) & ::pixelDetails::layerMask;
            module = (rawId >> ::pixelDetails::moduleStartBit) & ::pixelDetails::moduleMask;
            side = (module < 5) ? -1 : 1;
          } else {
            // endcap ids
            layer = 0;
            panel = (rawId >> ::pixelDetails::panelStartBit) & ::pixelDetails::panelMask;
            //disk  = (rawId >> diskStartBit_) & diskMask_;
            side = (panel == 1) ? -1 : 1;
            //blade = (rawId >> bladeStartBit_) & bladeMask_;
          }

          // ***special case of layer to 1 be handled here
          ::pixelDetails::Pixel localPix;
          if (layer == 1) {
            uint32_t col = (ww >> ::pixelDetails::COL_shift) & ::pixelDetails::COL_mask;
            uint32_t row = (ww >> ::pixelDetails::ROW_shift) & ::pixelDetails::ROW_mask;
            localPix.row = row;
            localPix.col = col;
            if (includeErrors) {
              if (not rocRowColIsValid(row, col)) {
                uint8_t error = conversionError(fedId, 3, debug);  //use the device function and fill the arrays
                err[gIndex].pixelErrors() = SiPixelErrorCompact{rawId, ww, error, fedId};
                alpaka::atomicInc(acc, &err.size(), 0xffffffff, alpaka::hierarchy::Threads{});
                if (debug)
                  printf("BPIX1  Error status: %i\n", error);
                return;
              }
            }
          } else {
            // ***conversion rules for dcol and pxid
            uint32_t dcol = (ww >> ::pixelDetails::DCOL_shift) & ::pixelDetails::DCOL_mask;
            uint32_t pxid = (ww >> ::pixelDetails::PXID_shift) & ::pixelDetails::PXID_mask;
            uint32_t row = ::pixelDetails::numRowsInRoc - pxid / 2;
            uint32_t col = dcol * 2 + pxid % 2;
            localPix.row = row;
            localPix.col = col;
            if (includeErrors and not dcolIsValid(dcol, pxid)) {
              uint8_t error = conversionError(fedId, 3, debug);
              err[gIndex].pixelErrors() = SiPixelErrorCompact{rawId, ww, error, fedId};
              alpaka::atomicInc(acc, &err.size(), 0xffffffff, alpaka::hierarchy::Threads{});
              if (debug)
                printf("Error status: %i %d %d %d %d\n", error, dcol, pxid, fedId, roc);
              return;
            }
          }

          ::pixelDetails::Pixel globalPix = frameConversion(barrel, side, layer, rocIdInDetUnit, localPix);
          dvgi.xx() = globalPix.row;  // origin shifting by 1 0-159
          dvgi.yy() = globalPix.col;  // origin shifting by 1 0-415
          dvgi.adc() = getADC(ww);
          dvgi.pdigi() = ::pixelDetails::pack(globalPix.row, globalPix.col, dvgi.adc());
          dvgi.moduleId() = detId.moduleId;
          dvgi.rawIdArr() = rawId;
        });  // end of stride on grid

      }  // end of Raw to Digi kernel operator()
    };   // end of Raw to Digi struct

    template <typename TrackerTraits>
    struct FillHitsModuleStart {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc &acc, SiPixelClustersSoAView clus_view) const {
        ALPAKA_ASSERT_OFFLOAD(TrackerTraits::numberOfModules < 2048);  // easy to extend at least till 32*1024

        constexpr int nMaxModules = TrackerTraits::numberOfModules;
        constexpr uint32_t maxHitsInModule = TrackerTraits::maxHitsInModule;

#ifndef NDEBUG
        [[maybe_unused]] const uint32_t blockIdxLocal(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
        ALPAKA_ASSERT_OFFLOAD(0 == blockIdxLocal);
        [[maybe_unused]] const uint32_t gridDimension(alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
        ALPAKA_ASSERT_OFFLOAD(1 == gridDimension);
#endif

        // limit to maxHitsInModule;
        cms::alpakatools::for_each_element_in_block_strided(acc, nMaxModules, [&](uint32_t i) {
          clus_view[i + 1].clusModuleStart() = std::min(maxHitsInModule, clus_view[i].clusInModule());
        });

        constexpr bool isPhase2 = std::is_base_of<pixelTopology::Phase2, TrackerTraits>::value;
        constexpr auto leftModules = isPhase2 ? 1024 : nMaxModules - 1024;

        auto &&ws = alpaka::declareSharedVar<uint32_t[32], __COUNTER__>(acc);

        cms::alpakatools::blockPrefixScan(
            acc, clus_view.clusModuleStart() + 1, clus_view.clusModuleStart() + 1, 1024, ws);

        cms::alpakatools::blockPrefixScan(
            acc, clus_view.clusModuleStart() + 1024 + 1, clus_view.clusModuleStart() + 1024 + 1, leftModules, ws);

        if constexpr (isPhase2) {
          cms::alpakatools::blockPrefixScan(
              acc, clus_view.clusModuleStart() + 2048 + 1, clus_view.clusModuleStart() + 2048 + 1, 1024, ws);
          cms::alpakatools::blockPrefixScan(acc,
                                            clus_view.clusModuleStart() + 3072 + 1,
                                            clus_view.clusModuleStart() + 3072 + 1,
                                            nMaxModules - 3072,
                                            ws);
        }

        constexpr auto lastModule = isPhase2 ? 2049u : nMaxModules + 1;
        cms::alpakatools::for_each_element_in_block_strided(acc, lastModule, 1025u, [&](uint32_t i) {
          clus_view[i].clusModuleStart() += clus_view[1024].clusModuleStart();
        });
        alpaka::syncBlockThreads(acc);

        if constexpr (isPhase2) {
          cms::alpakatools::for_each_element_in_block_strided(acc, 3073u, 2049u, [&](uint32_t i) {
            clus_view[i].clusModuleStart() += clus_view[2048].clusModuleStart();
          });
          alpaka::syncBlockThreads(acc);

          cms::alpakatools::for_each_element_in_block_strided(acc, nMaxModules + 1, 3073u, [&](uint32_t i) {
            clus_view[i].clusModuleStart() += clus_view[3072].clusModuleStart();
          });
          alpaka::syncBlockThreads(acc);
        }
#ifdef GPU_DEBUG
        ALPAKA_ASSERT_OFFLOAD(0 == clus_view[0].moduleStart());
        auto c0 = std::min(maxHitsInModule, clus_view[1].clusModuleStart());
        ALPAKA_ASSERT_OFFLOAD(c0 == clus_view[1].moduleStart());
        ALPAKA_ASSERT_OFFLOAD(clus_view[1024].moduleStart() >= clus_view[1023].moduleStart());
        ALPAKA_ASSERT_OFFLOAD(clus_view[1025].moduleStart() >= clus_view[1024].moduleStart());
        ALPAKA_ASSERT_OFFLOAD(clus_view[nMaxModules].moduleStart() >= clus_view[1025].moduleStart());

        cms::alpakatools::for_each_element_in_block_strided(acc, nMaxModules + 1, [&](uint32_t i) {
          if (0 != i)
            ALPAKA_ASSERT_OFFLOAD(clus_view[i].moduleStart() >= clus_view[i - i].moduleStart());
          // Check BPX2 (1), FP1 (4)
          constexpr auto bpix2 = TrackerTraits::layerStart[1];
          constexpr auto fpix1 = TrackerTraits::layerStart[4];
          if (i == bpix2 || i == fpix1)
            printf("moduleStart %d %d\n", i, clus_view[i].moduleStart());
        });
#endif
        // avoid overflow
        constexpr auto MAX_HITS = TrackerTraits::maxNumberOfHits;
        cms::alpakatools::for_each_element_in_block_strided(acc, nMaxModules + 1, [&](uint32_t i) {
          if (clus_view[i].clusModuleStart() > MAX_HITS)
            clus_view[i].clusModuleStart() = MAX_HITS;
        });

      }  // end of FillHitsModuleStart kernel operator()
    };   // end of FillHitsModuleStart struct

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
        alpaka::exec<Acc1D>(queue,
                            workDiv,
                            RawToDigi_kernel{},
                            cablingMap,
                            modToUnp,
                            wordCounter,
                            word_d.data(),
                            fedId_d.data(),
                            digis_d->view(),
                            digiErrors_d->view(),
                            useQualityInfo,
                            includeErrors,
                            debug);

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

        alpaka::exec<Acc1D>(
            queue, workDiv, CalibDigis{}, clusterThresholds, digis_d->view(), clusters_d->view(), gains, wordCounter);

#ifdef GPU_DEBUG
        alpaka::wait(queue);
        std::cout << "CountModules kernel launch with " << blocks << " blocks of " << threadsPerBlockOrElementsPerThread
                  << " threadsPerBlockOrElementsPerThread\n";
#endif

        alpaka::exec<Acc1D>(
            queue, workDiv, CountModules<TrackerTraits>{}, digis_d->view(), clusters_d->view(), wordCounter);

        auto moduleStartFirstElement =
            cms::alpakatools::make_device_view(alpaka::getDev(queue), clusters_d->view().moduleStart(), 1u);
        alpaka::memcpy(queue, nModules_Clusters_h, moduleStartFirstElement);

        // TODO
        // - we are fixing this here since it needs to be needed
        // at compile time also in the kernel (for_each_element_in_block_strided)
        // - put maxIter in the Geometry traits
        constexpr auto threadsOrElementsFindClus = 256;

        const auto workDivMaxNumModules =
            cms::alpakatools::make_workdiv<Acc1D>(numberOfModules, threadsOrElementsFindClus);
        // NB: With present FindClus() / chargeCut() algorithm,
        // threadPerBlock (GPU) or elementsPerThread (CPU) = 256 show optimal performance.
        // Though, it does not have to be the same number for CPU/GPU cases.

#ifdef GPU_DEBUG
        std::cout << " FindClus kernel launch with " << numberOfModules << " blocks of " << threadsOrElementsFindClus
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
        const auto clusModuleStartLastElement = cms::alpakatools::make_device_view(
            alpaka::getDev(queue), clusters_d->const_view().clusModuleStart() + numberOfModules, 1u);
        constexpr int startBPIX2 = TrackerTraits::layerStart[1];

        // element startBPIX2 hold the number of clusters until BPIX2
        const auto bpix2ClusterStart = cms::alpakatools::make_device_view(
            alpaka::getDev(queue), clusters_d->const_view().clusModuleStart() + startBPIX2, 1u);
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

      auto moduleStartFirstElement =
          cms::alpakatools::make_device_view(alpaka::getDev(queue), clusters_d->view().moduleStart(), 1u);
      alpaka::memcpy(queue, nModules_Clusters_h, moduleStartFirstElement);

      /// should be larger than maxPixInModule/16 aka (maxPixInModule/maxiter in the kernel)

      const auto threadsPerBlockFindClus = 256;
      const auto workDivMaxNumModules = cms::alpakatools::make_workdiv<Acc1D>(numberOfModules, threadsPerBlockFindClus);

#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "FindClus kernel launch with " << numberOfModules << " blocks of " << threadsPerBlockFindClus
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
      const auto clusModuleStartLastElement = cms::alpakatools::make_device_view(
          alpaka::getDev(queue), clusters_d->const_view().clusModuleStart() + numberOfModules, 1u);
      constexpr int startBPIX2 = pixelTopology::Phase2::layerStart[1];
      // element startBPIX2 hold the number of clusters until BPIX2
      const auto bpix2ClusterStart = cms::alpakatools::make_device_view(
          alpaka::getDev(queue), clusters_d->const_view().clusModuleStart() + startBPIX2, 1u);
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
