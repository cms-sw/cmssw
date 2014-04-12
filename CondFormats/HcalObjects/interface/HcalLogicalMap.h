#ifndef HcalLogicalMap_h
#define HcalLogicalMap_h

#include "CondFormats/HcalObjects/interface/HcalMappingEntry.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include <vector>
class HcalTopology;

class HcalLogicalMap {
  public:
         
  HcalLogicalMap(const HcalTopology*,
                   std::vector<HBHEHFLogicalMapEntry>&,
		   std::vector<HOHXLogicalMapEntry>&,
		   std::vector<CALIBLogicalMapEntry>&,
		   std::vector<ZDCLogicalMapEntry>&,
		   std::vector<HTLogicalMapEntry>&,
		   std::vector<uint32_t>&,
		   std::vector<uint32_t>&,
		   std::vector<uint32_t>&,
		   std::vector<uint32_t>&,
		   std::vector<uint32_t>&,
		   std::vector<uint32_t>&,
		   std::vector<uint32_t>&,
		   std::vector<uint32_t>&);

    ~HcalLogicalMap( );

    void checkHashIds();
    void checkElectronicsHashIds() ;
    void checkIdFunctions();
    void printMap( unsigned int mapIOV );
    HcalElectronicsMap generateHcalElectronicsMap();
    const DetId getDetId(const HcalElectronicsId&);
    const HcalFrontEndId getHcalFrontEndId(const DetId&);
    uint32_t static makeEntryNumber(bool,int,int);

  private:

    void printHBEFMap(  FILE* hbefmapfile );
    void printHOXMap(   FILE* hoxmapfile );
    void printCalibMap( FILE* calibmapfile );
    void printZDCMap(   FILE* zdcmapfile );
    void printHTMap(    FILE* htmapfile );

    unsigned int mapIOV_;

    std::vector<HBHEHFLogicalMapEntry> HBHEHFEntries_;
    std::vector<HOHXLogicalMapEntry>   HOHXEntries_;
    std::vector<CALIBLogicalMapEntry>  CALIBEntries_;
    std::vector<ZDCLogicalMapEntry>    ZDCEntries_;
    std::vector<HTLogicalMapEntry>     HTEntries_;
    std::vector<uint32_t> LinearIndex2Entry_;
    std::vector<uint32_t> HbHash2Entry_;
    std::vector<uint32_t> HeHash2Entry_;
    std::vector<uint32_t> HfHash2Entry_;
    std::vector<uint32_t> HtHash2Entry_;
    std::vector<uint32_t> HoHash2Entry_;
    std::vector<uint32_t> HxCalibHash2Entry_;
    std::vector<uint32_t> ZdcHash2Entry_;

    const HcalTopology* topo_;
};

#endif
