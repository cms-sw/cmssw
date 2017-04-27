/****************************************************************************
*
* This is a part of the TOTEM offline software.
* Authors: 
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#ifndef EventFilter_CTPPSRawToDigi_RawDataUnpacker
#define EventFilter_CTPPSRawToDigi_RawDataUnpacker

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/CTPPSDigi/interface/TotemFEDInfo.h"

#include "EventFilter/CTPPSRawToDigi/interface/VFATFrameCollection.h"
#include "EventFilter/CTPPSRawToDigi/interface/SimpleVFATFrameCollection.h"

//----------------------------------------------------------------------------------------------------

/// \brief Collection of code for unpacking of TOTEM raw-data.
class RawDataUnpacker
{
  public:
    typedef uint64_t word;

    /// VFAT transmission modes
    enum { vmCluster = 0x80, vmRaw = 0x90 };

    RawDataUnpacker() {}
    
    RawDataUnpacker(const edm::ParameterSet &conf);

    /// Unpack data from FED with fedId into `coll' collection.
    int Run(int fedId, const FEDRawData &data, std::vector<TotemFEDInfo> &fedInfoColl, SimpleVFATFrameCollection &coll) const;

    /// Process one Opto-Rx (or LoneG) frame.
    int ProcessOptoRxFrame(const word *buf, unsigned int frameSize, TotemFEDInfo &fedInfo, SimpleVFATFrameCollection *fc) const;

    /// Process one Opto-Rx frame in serial (old) format
    int ProcessOptoRxFrameSerial(const word *buffer, unsigned int frameSize, SimpleVFATFrameCollection *fc) const;

    /// Process one Opto-Rx frame in parallel (new) format
    int ProcessOptoRxFrameParallel(const word *buffer, unsigned int frameSize, TotemFEDInfo &fedInfo, SimpleVFATFrameCollection *fc) const;

    /// Process data from one VFAT in parallel (new) format
    int ProcessVFATDataParallel(const uint16_t *buf, unsigned int OptoRxId, SimpleVFATFrameCollection *fc) const;

  private:
    unsigned char verbosity;
};

#endif
