#ifndef RUNH4TABLEDAT_H
#define RUNH4TABLEDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class RunTableDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  RunTableDat();
  ~RunTableDat();

  // User data methods
  inline void setTableX(int num) { m_table_x = num; }
  inline int getTableX() const { return m_table_x; }

  inline void setTableY(int num) { m_table_y = num; }
  inline int getTableY() const { return m_table_y; }

  inline void setNumSpills(int num) { m_numSpills = num; }
  inline int getNumSpills() const { return m_numSpills; }

  inline void setNumEvents(int num) { m_numEvents = num; }
  inline int getNumEvents() const { return m_numEvents; }

 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const RunTableDat* item, RunIOV* iov )
    throw(std::runtime_error);

  void fetchData(std::map< EcalLogicID, RunTableDat >* fillMap, RunIOV* iov)
     throw(std::runtime_error);

  // User data
  int m_table_x ;
  int m_table_y ;
  int m_numSpills ;
  int m_numEvents;

};

#endif
