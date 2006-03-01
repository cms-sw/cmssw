#ifndef MONCRYSTALSTATUSDAT_H
#define MONCRYSTALSTATUSDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/MonCrystalStatusDef.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class MonCrystalStatusDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  MonCrystalStatusDat();
  ~MonCrystalStatusDat();

  // User data methods
  inline void setStatusG1(MonCrystalStatusDef status) { m_statusG1 = status; }
  inline MonCrystalStatusDef getStatusG1() const { return m_statusG1; }

  inline void setStatusG6(MonCrystalStatusDef status) { m_statusG6 = status; }
  inline MonCrystalStatusDef getStatusG6() const { return m_statusG6; }

  inline void setStatusG12(MonCrystalStatusDef status) { m_statusG12 = status; }
  inline MonCrystalStatusDef getStatusG12() const { return m_statusG12; }


 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const MonCrystalStatusDat* item, MonRunIOV* iov )
    throw(std::runtime_error);

  void fetchData(std::map< EcalLogicID, MonCrystalStatusDat >* fillMap, MonRunIOV* iov)
     throw(std::runtime_error);

  // User data
  MonCrystalStatusDef m_statusG1;
  MonCrystalStatusDef m_statusG6;
  MonCrystalStatusDef m_statusG12;
};

#endif
