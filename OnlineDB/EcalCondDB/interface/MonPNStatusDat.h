#ifndef MONPNSTATUSDAT_H
#define MONPNSTATUSDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/MonPNStatusDef.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class MonPNStatusDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  MonPNStatusDat();
  ~MonPNStatusDat();

  // User data methods
  inline void setStatusG1(MonPNStatusDef status) { m_statusG1 = status; }
  inline MonPNStatusDef getStatusG1() const { return m_statusG1; }

  inline void setStatusG16(MonPNStatusDef status) { m_statusG16 = status; }
  inline MonPNStatusDef getStatusG16() const { return m_statusG16; }

 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const MonPNStatusDat* item, MonRunIOV* iov )
    throw(std::runtime_error);

  void fetchData(std::map< EcalLogicID, MonPNStatusDat >* fillMap, MonRunIOV* iov)
     throw(std::runtime_error);

  // User data
  MonPNStatusDef m_statusG1;
  MonPNStatusDef m_statusG16;
};

#endif
