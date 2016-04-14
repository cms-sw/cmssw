/****************************************************************************
*
* This is a part of the TOTEM offline software.
* Authors: 
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#ifndef EventFilter_TotemRawToDigi_RawDataUnpacker
#define EventFilter_TotemRawToDigi_RawDataUnpacker

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include "EventFilter/TotemRawToDigi/interface/VFATFrameCollection.h"
#include "EventFilter/TotemRawToDigi/interface/SimpleVFATFrameCollection.h"

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
    int Run(int fedId, const FEDRawData &data, SimpleVFATFrameCollection &coll);

    /// Process one Opto-Rx (or LoneG) frame.
    int ProcessOptoRxFrame(word *buf, unsigned int frameSize, SimpleVFATFrameCollection *fc);

    /// Process one Opto-Rx frame in serial (old) format
    int ProcessOptoRxFrameSerial(word *buffer, unsigned int frameSize, SimpleVFATFrameCollection *fc);

    /// Process one Opto-Rx frame in parallel (new) format
    int ProcessOptoRxFrameParallel(word *buffer, unsigned int frameSize, SimpleVFATFrameCollection *fc);

    /// Process data from one VFAT in parallel (new) format
    int ProcessVFATDataParallel(unsigned short *buf, unsigned int OptoRxId, SimpleVFATFrameCollection *fc);
};

#endif
