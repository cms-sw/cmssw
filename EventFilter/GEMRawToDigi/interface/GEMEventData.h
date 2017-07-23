#ifndef GEMEventData_h
#define GEMEventData_h

class GEMCFEBData;
class CLCTData;
class TMBScope;
class GEMDMBHeader;
class GEMDMBTrailer;
class GEMStripDigi;
class GEMALCTHeader;
struct GEMALCTHeader2007;
class GEMAnodeData;
class GEMALCTTrailer;
class GEMTMBHeader;
class GEMTMBData;
class GEMCLCTData;
class GEMTMBTrailer;
class GEMWireDigi;
class GEMStripDigi;
class GEMComparatorOutput;
#include <map>
#include <vector>
#ifndef LOCAL_UNPACK
#include <atomic>
#endif
#include "EventFilter/GEMRawToDigi/interface/GEMDMBHeader.h"
#include "EventFilter/GEMRawToDigi/interface/GEMDMBTrailer.h"
#include "EventFilter/GEMRawToDigi/interface/GEMDMBHeader.h"
#include "EventFilter/GEMRawToDigi/interface/GEMDMBTrailer.h"
#include "EventFilter/GEMRawToDigi/interface/GEMALCTHeader.h"
#include "EventFilter/GEMRawToDigi/interface/GEMAnodeData.h"
#include "EventFilter/GEMRawToDigi/interface/GEMALCTTrailer.h"
#include "EventFilter/GEMRawToDigi/interface/GEMTMBData.h"
#include "EventFilter/GEMRawToDigi/interface/GEMDMBTrailer.h"
#include "DataFormats/GEMDigi/interface/GEMRPCDigi.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include <boost/dynamic_bitset.hpp>

/// Maximum available CFEBs per chamber (for old system 5, for new ME11 should be 7)
#define MAX_CFEB 7

class GEMEventData {
 public:
  explicit GEMEventData(int chamberType, uint16_t format_version = 2005);
  /// should make const input soon
  GEMEventData(unsigned short * buf, uint16_t format_version = 2005);
  GEMEventData(){}
  /// since we need deep copies, need the Big Three
  /// (destructor, copy ctor, op=)
  ~GEMEventData();
  GEMEventData(const GEMEventData & data);
  GEMEventData operator=(const GEMEventData & data);

  /// size of the data buffer used, in bytes
  unsigned short size() const {return size_;}

  /** turns on/off debug flag for this class */
  static void setDebug(const bool value) {debug = value;}


  ///if dealing with ALCT data
  bool isALCT(const short unsigned int * buf);

  ///if dealing with TMB data
  bool isTMB(const short unsigned int * buf);


  

  /// unpacked in long mode: has overflow and error bits decoded
  GEMCFEBData * cfebData(unsigned icfeb) const;

  /// returns all the strip digis in the chamber, with the comparator information.
  std::vector<GEMStripDigi> stripDigis(const GEMDetId & idlayer) const;

  /// returns all the strip digis in the chamber's cfeb
  std::vector<GEMStripDigi> stripDigis(unsigned idlayer, unsigned icfeb) const;


  /// deprecated.  Use the above methods instead
  std::vector< std::vector<GEMStripDigi> > stripDigis() const;
  

  std::vector<GEMWireDigi> wireDigis(unsigned ilayer) const;
  /// deprecated.  Use the above method instead.
  std::vector< std::vector<GEMWireDigi> > wireDigis() const;


  /// the flag for existence of ALCT data
  int nalct() const {return theDMBHeader.nalct();}

  /// the number of CLCTs
  int nclct() const {return theDMBHeader.nclct();}

  /// the DAQ motherboard header.  A good place for event and chamber info
  const GEMDMBHeader * dmbHeader() const {return &theDMBHeader;}
  GEMDMBHeader * dmbHeader()  {return &theDMBHeader;}
  
  /// user must check if nalct > 0
  GEMALCTHeader * alctHeader() const;

  /// user must check if nalct > 0
  GEMALCTTrailer * alctTrailer() const;

  /// user must check if nalct > 0
  GEMAnodeData * alctData() const;

  ///user must check in nclct > 0
  GEMTMBData * tmbData() const;

  /// user must check if nclct > 0
  GEMTMBHeader * tmbHeader() const;

  /// user must check if nclct > 0
  GEMCLCTData * clctData() const;

  /// DMB trailer
  const GEMDMBTrailer * dmbTrailer() const {return &theDMBTrailer;}
  /// routines to add digis to the data
  void add(const GEMStripDigi &, int layer);
  void add(const GEMWireDigi &, int layer);
  void add(const GEMComparatorDigi &, int layer);
  void add(const GEMComparatorDigi &, const GEMDetId &);
  /// these go in as vectors, so they get sorted right away
  void add(const std::vector<GEMALCTDigi> &);
  void add(const std::vector<GEMCLCTDigi> &);
  void add(const std::vector<GEMCorrelatedLCTDigi> &);

  
  /// this will fill the DMB header, and change all related fields in
  /// the DMBTrailer, ALCTHeader, and TMBHeader
  void setEventInformation(int bxnum, int lvl1num);

  /// returns the packed event data. 
  boost::dynamic_bitset<> pack();

  /// adds an empty ALCTHeader, trailer, and anode data
  void addALCTStructures();

  /// might not be set in real data
  int chamberType() const {return theChamberType;}

  uint16_t getFormatVersion() const { return theFormatVersion; }

  unsigned int calcALCTcrc(std::vector< std::pair<unsigned int, unsigned short*> > &vec);


#ifdef LOCAL_UNPACK
  static bool debug;
#else
  static std::atomic<bool> debug;
#endif
  //uint16_t dataPresent; // 7 bit word which will tell if alct, clct, and 5 cfebs are present
  static void selfTest();

private:
  /// helpers for ctors, dtor, and op=
  /// zeroes all pointers
  void init();
  void unpack_data(unsigned short * buf);
  void copy(const GEMEventData &);
  void destroy();

  /// makes new ALCT classes, if needed
  void checkALCTClasses();
  /// makes new TMB classes, if needed
  void checkTMBClasses();

  /// adds the comparators to the strip digis
  void addComparatorInformation(std::vector<GEMStripDigi>&, int layer) const;

  GEMDMBHeader theDMBHeader;
  //these are empty data objects unless filled in GEMEventData.cc
  /// these may or may not be present.  I decided to make them
  /// dynamic because most GEM chambers don't have LCTs,
  /// therefore don't have data, except for DMB headers and trailers.
  GEMALCTHeader * theALCTHeader;
  GEMAnodeData  * theAnodeData;
  GEMALCTTrailer * theALCTTrailer;
  GEMTMBData    * theTMBData;

  /// for up to MAX_CFEB CFEB boards
  GEMCFEBData * theCFEBData[MAX_CFEB];

  GEMDMBTrailer theDMBTrailer;

  int size_;
  /// this won't be filled when real data is read it.  It's only used when packing
  /// simulated data, so we know how many wire and strip channels to make.
  int theChamberType;
  
  /// Auxiliary bufer to recove the ALCT raw payload from zero suppression 
  unsigned short * alctZSErecovered;
  int zseEnable; 

  /// Output Format Version (2005, 2013)
  uint16_t theFormatVersion;
};

std::ostream & operator<<(std::ostream & os, const GEMEventData & evt);
#endif
