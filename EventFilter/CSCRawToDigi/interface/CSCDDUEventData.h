// Author Rick Wilkinson
//Modified 4/21/03 to store all CSC data in vectors
//A.Tumanov
//DDUHeader is separated into a class of its own

#ifndef CSCDDUEventData_h
#define CSCDDUEventData_h

#include <vector>
#include "EventFilter/CSCRawToDigi/interface/CSCEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDDUHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDDUTrailer.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDCCHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDCCTrailer.h"
#include <boost/dynamic_bitset.hpp>

class CSCDDUEventData {
public:

  explicit CSCDDUEventData(const CSCDDUHeader &);

  /// buf may need to stay pinned in memory as long
  /// as this data is used.  Not sure
  explicit CSCDDUEventData(unsigned short *buf);

  ~CSCDDUEventData();

  static void setDebug(bool value) {debug = value;} 
  static void setErrorMask(unsigned int value) {errMask = value;} 

  /// accessor to data
  const std::vector<CSCEventData> & cscData() const {return theData;}

  CSCDDUHeader header() const {return theDDUHeader;}
  CSCDDUTrailer trailer() const {return theDDUTrailer;}

  CSCDCCHeader dccHeader() const {return theDCCHeader;}
  CSCDCCTrailer dccTrailer() const {return theDCCTrailer;}


  /// for making events.  Sets the bxnum and lvl1num inside the chamber event
  void add(CSCEventData &);

  /// trailer info
  long unsigned int errorstat;

  bool check() const;

  /// prints out the error associated with this status 
  /// from the header or trailer
  void decodeStatus(int status) const;
  void decodeStatus() const;
  int sizeInWords() const {return theSizeInWords;}
  int size() const {return theSizeInWords*16;} ///Alex check this 16 or 64

  /// returns packed event data
  boost::dynamic_bitset<> pack();

  
  static bool debug;
  static unsigned int errMask;

  /// a good test routine would be to unpack data, then pack it again.
protected:
  void unpack_data(unsigned short * buf);
  CSCDCCHeader theDCCHeader;
  CSCDDUHeader theDDUHeader;
  // CSCData is unpacked and stored in this vector
  std::vector<CSCEventData> theData;
  CSCDDUTrailer theDDUTrailer;
  CSCDCCTrailer theDCCTrailer;
  int theSizeInWords;
};

#endif
