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

#include "CondFormats/PPSObjects/interface/TotemDAQMapping.h"

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

//brief Collection of code to convert TOTEM raw data into digi.

class FEDRawData;
class CTPPSTotemDigiToRaw;

class CTPPSTotemDataFormatter {
private:
  typedef uint16_t Word16;
  typedef uint32_t Word32;
  typedef uint64_t Word64;

  int m_WordCounter;
  int m_DigiCounter;

public:
  typedef std::unordered_map<int, FEDRawData> RawData;
  typedef std::vector<TotemRPDigi> DetDigis;
  typedef std::unordered_map<cms_uint32_t, DetDigis> Digis;

  CTPPSTotemDataFormatter(std::map<TotemFramePosition, TotemVFATInfo> const& mapping);

  int nWords() const { return m_WordCounter; }
  int nDigis() const { return m_DigiCounter; }

  struct PPSStripIndex {
    uint32_t id;
    unsigned int hwid;
    short unsigned int fedid;
    short unsigned int idxinfiber;
    short unsigned int gohid;
  };

  void formatRawData(unsigned int lvl1_ID,
                     RawData& fedRawData,
                     const Digis& digis,
                     std::vector<PPSStripIndex> v_iDdet2fed);

  static bool compare(const PPSStripIndex& a, const PPSStripIndex& b) { return a.id < b.id; }

  std::string print(const Word64& word) const;
};

#endif
