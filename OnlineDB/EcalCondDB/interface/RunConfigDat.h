#ifndef RUNCONFIGDAT_H
#define RUNCONFIGDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class RunConfigDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  RunConfigDat();
  ~RunConfigDat();

  // User data methods
  inline std::string getTable() { return "RUN_CONFIG_DAT"; }

  inline std::string getConfigTag() const { return m_configTag; }
  inline void setConfigTag(std::string tag) { m_configTag = tag; }

  inline int getConfigVersion() const { return m_configVer; }
  inline void setConfigVersion(int ver) { m_configVer = ver; }

 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const RunConfigDat* item, RunIOV* iov )
    throw(std::runtime_error);

  void fetchData(std::map< EcalLogicID, RunConfigDat >* fillMap, RunIOV* iov)
     throw(std::runtime_error);

  // User data
  std::string m_configTag;
  int m_configVer;

};

#endif
