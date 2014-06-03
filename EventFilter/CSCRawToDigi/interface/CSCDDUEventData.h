#ifndef CSCDDUEventData_h
#define CSCDDUEventData_h

/** \class CSCDDUEventData
 *
 * \author Rick Wilkinson
 * \author A. Tumanov
 */

#include <vector>
#include "EventFilter/CSCRawToDigi/interface/CSCEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDDUHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDDUTrailer.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDCCHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDCCTrailer.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDCCExaminer.h"
#include <boost/dynamic_bitset.hpp>

class CSCDDUEventData {
public:

  explicit CSCDDUEventData(const CSCDDUHeader &);

  // buf may need to stay pinned in memory as long
  // as this data is used.  Not sure
  explicit CSCDDUEventData(unsigned short *buf, CSCDCCExaminer* examiner=NULL);

  ~CSCDDUEventData();

  static void setDebug(bool value) {debug = value;} 
  static void setErrorMask(unsigned int value) {errMask = value;} 

  /// accessor to data
  const std::vector<CSCEventData> & cscData() const {return theData;}

  CSCDDUHeader header() const {return theDDUHeader;}
  CSCDDUTrailer trailer() const {return theDDUTrailer;}
  uint16_t trailer0() const {return theDDUTrailer0;}

  CSCDCCHeader dccHeader() const {return theDCCHeader;}
  CSCDCCTrailer dccTrailer() const {return theDCCTrailer;}


  /// for making events.  Sets the bxnum and lvl1num inside the chamber event
    void add(CSCEventData &, int dmbId, int dduInput, unsigned int format_version = 2005);

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

#ifdef LOCAL_UNPACK
  static bool debug;
#else
  static std::atomic<bool> debug;
#endif
  static unsigned int errMask;

  /// a good test routine would be to unpack data, then pack it again.
protected:
  void unpack_data(unsigned short * buf, CSCDCCExaminer* examiner=NULL);
  CSCDCCHeader theDCCHeader;
  CSCDDUHeader theDDUHeader;
  // CSCData is unpacked and stored in this vector
  std::vector<CSCEventData> theData;
  CSCDDUTrailer theDDUTrailer;
  CSCDCCTrailer theDCCTrailer;
  uint16_t theDDUTrailer0;
  int theSizeInWords;
  uint16_t theFormatVersion;
};

#endif
