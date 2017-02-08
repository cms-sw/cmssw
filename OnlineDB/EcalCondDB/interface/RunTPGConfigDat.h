#ifndef RUNTPGCONFIGDAT_H
#define RUNTPGCONFIGDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class RunTPGConfigDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  RunTPGConfigDat();
  ~RunTPGConfigDat();

  // User data methods
  inline std::string getTable() { return "RUN_TPGCONFIG_DAT"; }

  inline std::string getConfigTag() const { return m_config; }
  inline void setConfigTag(std::string x) { m_config = x; }
  inline int getVersion() const { return m_version; }
  inline void setVersion(int x) { m_version = x; }


 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const RunTPGConfigDat* item, RunIOV* iov )
    throw(std::runtime_error);

  void fetchData(std::map< EcalLogicID, RunTPGConfigDat >* fillMap, RunIOV* iov)
     throw(std::runtime_error);

  // User data
  int m_version;
  std::string m_config;

};

#endif
