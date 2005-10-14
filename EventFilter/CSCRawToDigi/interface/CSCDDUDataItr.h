#ifndef CSCDDUDataItr_h
#define CSCDDUDataItr_h

/** a class to help users iterate over CSC chambers,
    without having to know about DDUs
    \Author Rick Wilkinson, Caltech
*/

class CSCDDUEventData;
class CSCEventData;
#include<vector>

class CSCDDUDataItr {
public:
  /// default constructor
  CSCDDUDataItr();

  /// construct from data buffer. so makes a new DDUEventData
  CSCDDUDataItr(const char * buf);

  /// uses someone else's data, so doesn't delete
  CSCDDUDataItr(const CSCDDUEventData * dduData);

  ~CSCDDUDataItr();

  /// if I own the data, I need to do special copy & assign
  CSCDDUDataItr(const CSCDDUDataItr &);
  void operator=(const CSCDDUDataItr &);

  bool next();

  const CSCEventData & operator*();

private:

  const CSCDDUEventData * theDDUData;
  int theCurrentCSC;
  int theNumberOfCSCs;
  bool theDataIsOwnedByMe;
};

#endif

