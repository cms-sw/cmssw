#ifndef RUNFECONFIGDAT_H
#define RUNFECONFIGDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class RunFEConfigDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  RunFEConfigDat();
  ~RunFEConfigDat();

  // User data methods
  inline std::string getTable() { return "RUN_FECONFIG_DAT"; }

  inline int getConfigId() const { return m_config; }
  inline void setConfigId(int x) { m_config = x; }


 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const RunFEConfigDat* item, RunIOV* iov )
    throw(std::runtime_error);

  void fetchData(std::map< EcalLogicID, RunFEConfigDat >* fillMap, RunIOV* iov)
     throw(std::runtime_error);

  // User data

  int m_config;

};

#endif
