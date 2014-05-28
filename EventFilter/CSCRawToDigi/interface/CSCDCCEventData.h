/// 01/20/05 A.Tumanov

#ifndef CSCDCCEventData_h
#define CSCDCCEventData_h

#include <vector>
#include <boost/dynamic_bitset.hpp>
#include "EventFilter/CSCRawToDigi/interface/CSCDDUEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDCCHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDCCTrailer.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDCCExaminer.h"

class CSCDCCEventData {
public:
  CSCDCCEventData(int sourceId, int nDDUs, int bx, int l1a);
  /// buf may need to stay pinned in memory as long
  /// as this data is used.  Not sure
  explicit CSCDCCEventData(unsigned short *buf, CSCDCCExaminer* examiner=NULL);

  ~CSCDCCEventData();

  static void setDebug(bool value) {debug = value;} 
 
  /// accessor to dduData
  const std::vector<CSCDDUEventData> & dduData() const {return theDDUData;}
  std::vector<CSCDDUEventData> & dduData() {return theDDUData;}

  CSCDCCHeader dccHeader() const {return theDCCHeader;}
  CSCDCCTrailer dccTrailer() const {return theDCCTrailer;}


  /// for making events.  Sets the bxnum and lvl1num inside the chamber event
  //void add(CSCEventData &);

  bool check() const;

  /// prints out the error associated with this status 
  /// from the header or trailer
  int sizeInWords() const {return theSizeInWords;}

  void addChamber(CSCEventData & chamber, int dduID, int dduSlot, int dduInput, int dmbID, uint16_t format_version = 2005);

  ///packs data into bits
  boost::dynamic_bitset<> pack();  

#ifdef LOCAL_UNPACK
  static bool debug;
#else
  static std::atomic<bool> debug;  
#endif


protected:
  void unpack_data(unsigned short * buf, CSCDCCExaminer* examiner=NULL);
  CSCDCCHeader theDCCHeader;
  // DDUData is unpacked and stored in this vector
  std::vector<CSCDDUEventData> theDDUData;
  CSCDCCTrailer theDCCTrailer;
  int theSizeInWords;
  
};

#endif
