#ifndef CSCEventData_h
#define CSCEventData_h

class CSCCFEBData;
class CLCTData;
class TMBScope;
class CSCDMBHeader;
class CSCDMBTrailer;
class CSCStripDigi;
class CSCALCTHeader;
class CSCAnodeData;
class CSCALCTTrailer;
class CSCTMBHeader;
class CSCTMBData;
class CSCCLCTData;
class CSCTMBTrailer;
class CSCWireDigi;
class CSCStripDigi;
class CSCComparatorOutput;
class BitVector;
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
#ifndef UNPCK_ONLY
#include "DataFormats/CSCDigi/interface/CSCRPCDigi.h"
#endif


class CSCEventData {
 public:
  explicit CSCEventData(int chamberType);
  /// should make const input soon
  CSCEventData(unsigned short * buf);
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

  /// unpacked in long mode: has overflow and error bits decoded
  CSCCFEBData * cfebData(unsigned icfeb) const;

  /// returns all the strip digis in the chamber, with the comparator information.
  std::vector<CSCStripDigi> stripDigis(unsigned ilayer) const;

  /// returns all the strip digis in the chamber's cfeb
  std::vector<CSCStripDigi> stripDigis(unsigned idlayer, unsigned icfeb) const;


  /// deprecated.  Use the above methods instead
  std::vector< std::vector<CSCStripDigi> > stripDigis() const;
  

  std::vector<CSCWireDigi> wireDigis(unsigned ilayer) const;
  /// deprecated.  Use the above method instead.
  std::vector< std::vector<CSCWireDigi> > wireDigis() const;


  /// the number of ALCT's
  int nalct() const {return nalct_;}

  /// the number of CLCTs
  int nclct() const {return nclct_;}

  /// the DAQ motherboard header.  A good place for event and chamber info
  const CSCDMBHeader & dmbHeader() const {return theDMBHeader;}
  CSCDMBHeader & dmbHeader()  {return theDMBHeader;}
  
  /// user must check if nalct > 0
  CSCALCTHeader alctHeader() const;

  /// user must check if nalct > 0
  CSCALCTTrailer alctTrailer() const;

  /// user must check if nalct > 0
  CSCAnodeData & alctData() const;

  ///user must check in nclct > 0
  CSCTMBData & tmbData() const;

  /// user must check if nclct > 0
  CSCTMBHeader & tmbHeader() const;

  /// user must check if nclct > 0
  CSCCLCTData & clctData() const;

  /// DMB trailer
  CSCDMBTrailer dmbTrailer() const {return theDMBTrailer;}
  /// routines to add digis to the data
  void add(const CSCStripDigi &, int layer);
  void add(const CSCWireDigi &, int layer);
  void add(const CSCComparatorOutput &, int layer);
  
  /// this will fill the DMB header, and change all related fields in
  /// the DMBTrailer, ALCTHeader, and TMBHeader
  void setEventInformation(int bxnum, int lvl1num);

  /// returns the packed event data. 
  std::pair<int, unsigned short *> pack();
  BitVector packVector();

  /// adds an empty ALCTHeader, trailer, and anode data
  void addALCTStructures();

  /// might not be set in real data
  int chamberType() const {return theChamberType;}

  static bool debug;

private:
  /// helpers for ctors, dtor, and op=
  /// zeroes all pointers
  void init();
  void copy(const CSCEventData &);
  void destroy();

  /// makes new ALCT classes
  void createALCTClasses();
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

  /// for up to 5 CFEB boards
  CSCCFEBData * theCFEBData[5];

  CSCDMBTrailer theDMBTrailer;

  int nalct_;
  int nclct_;
  int size_;
  /// this won't be filled when real data is read it.  It's only used when packing
  /// simulated data, so we know how many wire and strip channels to make.
  int theChamberType;
};

std::ostream & operator<<(std::ostream & os, const CSCEventData & evt);
#endif
