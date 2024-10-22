#ifndef RUNFECONFIGDAT_H
#define RUNFECONFIGDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/ODDelaysDat.h"

class RunFEConfigDat : public IDataItem {
public:
  friend class EcalCondDBInterface;
  RunFEConfigDat();
  ~RunFEConfigDat() override;

  // User data methods
  inline std::string getTable() override { return "RUN_FECONFIG_DAT"; }

  inline int getConfigId() const { return m_config; }
  inline void setConfigId(int x) { m_config = x; }

  std::list<ODDelaysDat> getDelays();

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const RunFEConfigDat* item, RunIOV* iov) noexcept(false);

  void fetchData(std::map<EcalLogicID, RunFEConfigDat>* fillMap, RunIOV* iov) noexcept(false);

  // User data

  int m_config;
};

#endif
