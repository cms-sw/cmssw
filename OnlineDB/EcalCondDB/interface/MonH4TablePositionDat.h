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
  ~MonH4TablePositionDat() override;

  // User data methods
  inline std::string getTable() override { return "MON_H4_TABLE_POSITION_DAT"; }

  inline void setTableX(float x) { m_tableX = x; }
  inline float getTableX() const { return m_tableX; }

  inline void setTableY(float y) { m_tableY = y; }
  inline float getTableY() const { return m_tableY; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const MonH4TablePositionDat* item, MonRunIOV* iov) noexcept(false);

  void fetchData(std::map<EcalLogicID, MonH4TablePositionDat>* fillMap, MonRunIOV* iov) noexcept(false);

  void writeArrayDB(const std::map<EcalLogicID, MonH4TablePositionDat>* data, MonRunIOV* iov) noexcept(false);

  // User data
  float m_tableX;
  float m_tableY;
};

#endif
