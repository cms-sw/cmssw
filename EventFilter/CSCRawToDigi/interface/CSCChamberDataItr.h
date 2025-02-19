#ifndef CSCChamberDataItr_h
#define CSCChamberDataItr_h

/** a class to help users iterate over CSC chambers,
    without having to know about DCCs and DDUs
    \Author Rick Wilkinson, Caltech
*/

#include "EventFilter/CSCRawToDigi/interface/CSCDDUDataItr.h"
class CSCDCCEventData;
class CSCDDUEventData;
class CSCEventData;
#include<vector>

class CSCChamberDataItr {
public:
  /// construct from data buffer.  Will figure out whether it's
  /// DCC or DDU
  CSCChamberDataItr(const char * buf);
  ~CSCChamberDataItr();

  bool next();

  const CSCEventData & operator*();

private:
  /// for DCC data.
  void constructFromDCC(const CSCDCCEventData &);
  /// for DDU-only data
  void constructFromDDU(const CSCDDUEventData &);

  /// sets theDDU & theNumberOfCSCs
  void initDDU();

  /// a little confusing here.  This class will either
  /// own theDCCData, in which case the DDUs points inside it,
  //  or if there's no DCC, it will
  /// make a new vector of DDUs (length 1).
  const CSCDCCEventData * theDCCData;
  CSCDDUDataItr * theDDUItr;
  unsigned theCurrentDDU;
  unsigned theNumberOfDDUs;
};

#endif

