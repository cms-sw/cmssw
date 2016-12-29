#ifndef HcalLogicalMap_h
#define HcalLogicalMap_h

#include "CondFormats/HcalObjects/interface/HcalMappingEntry.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include <vector>

////special for xml generation
//#include "CalibCalorimetry/HcalTPGAlgos/interface/HCALLMAPXMLDOMBlock.h"
//#include "CalibCalorimetry/HcalTPGAlgos/interface/HCALLMAPXMLProcessor.h"
////#include "CondFormats/HcalObjects/interface/HCALLMAPXMLDOMBlock.h"
////#include "CondFormats/HcalObjects/interface/HCALLMAPXMLProcessor.h"

class HcalTopology;

class HcalLogicalMap 
{
  public:
         
    HcalLogicalMap(const HcalTopology*,
                   std::vector<HBHEHFLogicalMapEntry>&,
		   std::vector<HOHXLogicalMapEntry>&,
	           std::vector<CALIBLogicalMapEntry>&,
         	   std::vector<ZDCLogicalMapEntry>&,
		   std::vector<HTLogicalMapEntry>&,
                   std::vector<OfflineDB>&,
                   std::vector<QIEMap>&,
		   std::vector<uint32_t>&,
		   std::vector<uint32_t>&,
		   std::vector<uint32_t>&,
		   std::vector<uint32_t>&,
		   std::vector<uint32_t>&,
		   std::vector<uint32_t>&,
		   std::vector<uint32_t>&,
		   std::vector<uint32_t>&);

    ~HcalLogicalMap();

    void checkHashIds();
    void checkElectronicsHashIds() ;
    void checkIdFunctions();
    //this printHTRLMap function is for the old HTR logical map txt file output, comment from hua.wei@cern.ch
    void printHTRLMap( unsigned int mapIOV );
    //print map with micro TCA backend, added by hua.wei@cern.ch
    void printuHTRLMap( unsigned int mapIOV);
    //print offline DB, added by hua.wei@cern.ch
    void printOfflineDB( unsigned int mapIOV );
    //print QIE map, added by hua.wei@cern.ch
    void printQIEMap( unsigned int mapIOV );
    //new for xml generation
    //void printXMLTables(unsigned int mapIOV);

    //generate electronics map
    HcalElectronicsMap generateHcalElectronicsMap();
    void printuHTREMap(std::ostream& fOutput);

    const DetId getDetId(const HcalElectronicsId&);
    const HcalFrontEndId getHcalFrontEndId(const DetId&);
    uint32_t static makeEntryNumber(bool,int,int);

  private:

    void printHBEFMap(  FILE* hbefmapfile );
    void printHOXMap(   FILE* hoxmapfile );
    void printCalibMap( FILE* calibmapfile );
    void printZDCMap(   FILE* zdcmapfile );
    void printHTMap(    FILE* htmapfile );

    void printuHTRHBEFMap(FILE* hbefuhtrmapfile);
    void printHCALOfflineDB(FILE* hcalofflinedb);
    void printHCALQIEMap(FILE* hcalqiemap);

    ////new for xml generation
    //void printHBEFXML(  unsigned int mapIOV );
    //void printHOXXML(   unsigned int mapIOV );
    //void printCalibXML( unsigned int mapIOV );
    //void printZDCXML(   unsigned int mapIOV );
    //void printHTXML(    unsigned int mapIOV );

    unsigned int mapIOV_;

  public:
    std::vector<HBHEHFLogicalMapEntry> HBHEHFEntries_;
    std::vector<HOHXLogicalMapEntry>   HOHXEntries_;
    std::vector<CALIBLogicalMapEntry>  CALIBEntries_;
    std::vector<ZDCLogicalMapEntry>    ZDCEntries_;
    std::vector<HTLogicalMapEntry>     HTEntries_;
    std::vector<OfflineDB>             OfflineDatabase_;
    std::vector<QIEMap>                QIEMaps_;

  private:
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
