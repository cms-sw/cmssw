/****************************************************************************
*
* This is a part of the TOTEM offline software.
* Authors:
*   Jan Kašpar (jan.kaspar@gmail.com)
*   Nicola Minafra
*
****************************************************************************/

#ifndef EventFilter_CTPPSRawToDigi_RawDataUnpacker
#define EventFilter_CTPPSRawToDigi_RawDataUnpacker

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/CTPPSDigi/interface/TotemFEDInfo.h"

#include "EventFilter/CTPPSRawToDigi/interface/VFATFrameCollection.h"
#include "EventFilter/CTPPSRawToDigi/interface/SimpleVFATFrameCollection.h"
#include "CondFormats/PPSObjects/interface/TotemT2FramePosition.h"

namespace pps {
  /// \brief Collection of code for unpacking of TOTEM raw-data.
  class RawDataUnpacker {
  public:
    typedef uint64_t word;

    /// VFAT transmission modes
    enum { vmCluster = 0x80, vmRaw = 0x90, vmDiamondCompact = 0xB0 };

    // list of headers for all words encountered in diamond data frames
    static constexpr unsigned int VFAT_DIAMOND_HEADER_OF_WORD_2 = 0x7800;
    static constexpr unsigned int VFAT_DIAMOND_HEADER_OF_WORD_3 = 0x7000;
    static constexpr unsigned int VFAT_DIAMOND_HEADER_OF_WORD_5 = 0x6800;
    static constexpr unsigned int VFAT_DIAMOND_HEADER_OF_WORD_7 = 0x6000;
    static constexpr unsigned int VFAT_HEADER_OF_EC = 0xC000;

    RawDataUnpacker() {}

    RawDataUnpacker(const edm::ParameterSet &conf);

    /// Unpack data from FED with fedId into `coll' collection.
    int run(int fedId,
            const FEDRawData &data,
            std::vector<TotemFEDInfo> &fedInfoColl,
            SimpleVFATFrameCollection &coll) const;

    /// Process one Opto-Rx (or LoneG) frame.
    int processOptoRxFrame(const word *buf,
                           unsigned int frameSize,
                           TotemFEDInfo &fedInfo,
                           SimpleVFATFrameCollection *fc) const;

    /// Process one Opto-Rx frame in serial (old) format
    int processOptoRxFrameSerial(const word *buffer, unsigned int frameSize, SimpleVFATFrameCollection *fc) const;

    /// Process one Opto-Rx frame in parallel (new) format
    int processOptoRxFrameParallel(const word *buffer,
                                   unsigned int frameSize,
                                   TotemFEDInfo &fedInfo,
                                   SimpleVFATFrameCollection *fc) const;

    /// Process one Opto-Rx frame that contains SAMPIC frames
    int processOptoRxFrameSampic(const word *buffer,
                                 unsigned int frameSize,
                                 TotemFEDInfo &fedInfo,
                                 SimpleVFATFrameCollection *fc) const;

    /// Process data from one VFAT in parallel (new) format
    int processVFATDataParallel(const uint16_t *buf,
                                unsigned int maxWords,
                                unsigned int OptoRxId,
                                SimpleVFATFrameCollection *fc) const;

  private:
    unsigned char verbosity;
  };
}  // namespace pps

#endif
