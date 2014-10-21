#ifndef CSCEventData_h
#define CSCEventData_h

class CSCCFEBData;
class CLCTData;
class TMBScope;
class CSCDMBHeader;
class CSCDMBTrailer;
class CSCStripDigi;
class CSCALCTHeader;
class CSCALCTHeader2007;
class CSCAnodeData;
class CSCALCTTrailer;
class CSCTMBHeader;
class CSCTMBData;
class CSCCLCTData;
class CSCTMBTrailer;
class CSCWireDigi;
class CSCStripDigi;
class CSCComparatorOutput;
#include <map>
#include <vector>
#include "EventFilter/CSCRawToDigi/interface/CSCDMBHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBTrailer.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBTrailer.h"
#include "EventFilter/CSCRawToDigi/interface/CSCALCTHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCAnodeData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCALCTTrailer.h"
#include "EventFilter/CSCRawToDigi/interface/CSCTMBData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBTrailer.h"
#include "DataFormats/CSCDigi/interface/CSCRPCDigi.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include <boost/dynamic_bitset.hpp>

/// Maximum available CFEBs per chamber (for old system 5, for new ME11 should be 7)
#define MAX_CFEB 7

class CSCEventData {
 public:
  explicit CSCEventData(int chamberType, uint16_t format_version = 2005);
  /// should make const input soon
  CSCEventData(unsigned short * buf, uint16_t format_version = 2005);
  CSCEventData(){}
  /// since we need deep copies, need the Big Three
  /// (destructor, copy ctor, op=)
  ~CSCEventData();
  CSCEventData(const CSCEventData & data);
  CSCEventData operator=(const CSCEventData & data);

  /// size of the data buffer used, in bytes
  unsigned short size() const {return size_;}

  /** turns on/off debug flag for this class */
  static void setDebug(const bool value) {debug = value;}


  ///if dealing with ALCT data
  bool isALCT(const short unsigned int * buf);

  ///if dealing with TMB data
  bool isTMB(const short unsigned int * buf);


  

  /// unpacked in long mode: has overflow and error bits decoded
  CSCCFEBData * cfebData(unsigned icfeb) const;

  /// returns all the strip digis in the chamber, with the comparator information.
  std::vector<CSCStripDigi> stripDigis(const CSCDetId & idlayer) const;

  /// returns all the strip digis in the chamber's cfeb
  std::vector<CSCStripDigi> stripDigis(unsigned idlayer, unsigned icfeb) const;


  /// deprecated.  Use the above methods instead
  std::vector< std::vector<CSCStripDigi> > stripDigis() const;
  

  std::vector<CSCWireDigi> wireDigis(unsigned ilayer) const;
  /// deprecated.  Use the above method instead.
  std::vector< std::vector<CSCWireDigi> > wireDigis() const;


  /// the flag for existence of ALCT data
  int nalct() const {return theDMBHeader.nalct();}

  /// the number of CLCTs
  int nclct() const {return theDMBHeader.nclct();}

  /// the DAQ motherboard header.  A good place for event and chamber info
  const CSCDMBHeader * dmbHeader() const {return &theDMBHeader;}
  CSCDMBHeader * dmbHeader()  {return &theDMBHeader;}
  
  /// user must check if nalct > 0
  CSCALCTHeader * alctHeader() const;

  /// user must check if nalct > 0
  CSCALCTTrailer * alctTrailer() const;

  /// user must check if nalct > 0
  CSCAnodeData * alctData() const;

  ///user must check in nclct > 0
  CSCTMBData * tmbData() const;

  /// user must check if nclct > 0
  CSCTMBHeader * tmbHeader() const;

  /// user must check if nclct > 0
  CSCCLCTData * clctData() const;

  /// DMB trailer
  const CSCDMBTrailer * dmbTrailer() const {return &theDMBTrailer;}
  /// routines to add digis to the data
  void add(const CSCStripDigi &, int layer);
  void add(const CSCWireDigi &, int layer);
  void add(const CSCComparatorDigi &, int layer);
  void add(const CSCComparatorDigi &, const CSCDetId &);
  /// these go in as vectors, so they get sorted right away
  void add(const std::vector<CSCALCTDigi> &);
  void add(const std::vector<CSCCLCTDigi> &);
  void add(const std::vector<CSCCorrelatedLCTDigi> &);

  
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


  static bool debug;
  //uint16_t dataPresent; // 7 bit word which will tell if alct, clct, and 5 cfebs are present
  static void selfTest();

private:
  /// helpers for ctors, dtor, and op=
  /// zeroes all pointers
  void init();
  void unpack_data(unsigned short * buf);
  void copy(const CSCEventData &);
  void destroy();

  /// makes new ALCT classes, if needed
  void checkALCTClasses();
  /// makes new TMB classes, if needed
  void checkTMBClasses();

  /// adds the comparators to the strip digis
  void addComparatorInformation(std::vector<CSCStripDigi>&, int layer) const;

  CSCDMBHeader theDMBHeader;
  //these are empty data objects unless filled in CSCEventData.cc
  /// these may or may not be present.  I decided to make them
  /// dynamic because most CSC chambers don't have LCTs,
  /// therefore don't have data, except for DMB headers and trailers.
  CSCALCTHeader * theALCTHeader;
  CSCAnodeData  * theAnodeData;
  CSCALCTTrailer * theALCTTrailer;
  CSCTMBData    * theTMBData;

  /// for up to MAX_CFEB CFEB boards
  CSCCFEBData * theCFEBData[MAX_CFEB];

  CSCDMBTrailer theDMBTrailer;

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

std::ostream & operator<<(std::ostream & os, const CSCEventData & evt);
#endif
