#ifndef MONH4TABLEPOSITIONDAT_H
#define MONH4TABLEPOSITIONDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class MonH4TablePositionDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  MonH4TablePositionDat();
  ~MonH4TablePositionDat();

  // User data methods
  inline std::string getTable() { return "MON_H4_TABLE_POSITION_DAT"; }

  inline void setTableX(float x) { m_tableX = x; }
  inline float getTableX() const { return m_tableX; }

  inline void setTableY(float y) { m_tableY = y; }
  inline float getTableY() const { return m_tableY; }

 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const MonH4TablePositionDat* item, MonRunIOV* iov )
    throw(std::runtime_error);

  void fetchData(std::map< EcalLogicID, MonH4TablePositionDat >* fillMap, MonRunIOV* iov)
     throw(std::runtime_error);

  void writeArrayDB(const std::map< EcalLogicID, MonH4TablePositionDat >* data, MonRunIOV* iov)
    throw(std::runtime_error);

  // User data
  float m_tableX;
  float m_tableY;
};

#endif
