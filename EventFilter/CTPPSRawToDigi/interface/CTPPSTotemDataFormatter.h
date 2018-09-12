/****************************************************************************
 *
 * This is a part of the TOTEM offline software.
 * Authors: 
 *
 ****************************************************************************/

#ifndef EventFilter_CTPPSRawToDigi_CTPPSTotemDataFormatter_h
#define EventFilter_CTPPSRawToDigi_CTPPSTotemDataFormatter_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include "CondFormats/CTPPSReadoutObjects/interface/TotemDAQMapping.h"

#include "DataFormats/CTPPSDigi/interface/TotemRPDigi.h"
#include "DataFormats/CTPPSDigi/interface/TotemVFATStatus.h"
#include "DataFormats/CTPPSDigi/interface/CTPPSDiamondDigi.h"

#include "FWCore/Utilities/interface/typedefs.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include <cstdint>
#include <vector>
#include <map>
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/CTPPSDigi/interface/TotemVFATStatus.h"
#include "EventFilter/CTPPSRawToDigi/interface/VFATFrame.h"
#include "FWCore/Framework/interface/EventSetup.h"
/// \brief Collection of code to convert TOTEM raw data into digi.

class FEDRawData;
class CTPPSTotemDigiToRaw;  

class CTPPSTotemDataFormatter {

  private:
    typedef uint16_t Word16;
    typedef uint32_t Word32;
    typedef uint64_t Word64;

    mutable int theWordCounter;
    mutable int theDigiCounter;

  public:

    typedef std::map<int, FEDRawData> RawData;
    typedef std::vector<TotemRPDigi> DetDigis;
    typedef std::map<cms_uint32_t,DetDigis> Digis;

    CTPPSTotemDataFormatter(std::map<TotemFramePosition, TotemVFATInfo> const &mapping);

    int nWords() const { return theWordCounter; }
    int nDigis() const { return theDigiCounter; }

     void formatRawData(unsigned int lvl1_ID, RawData & fedRawData, const Digis & digis, std::map<std::map<const uint32_t, unsigned int>, std::map<short unsigned int, std::map<short unsigned int, short unsigned int>>> iDdet2fed ) ; 

     std::string print(const Word64 & word) const;
};

#endif
