#ifndef RUNLASERRUNDAT_H
#define RUNLASERRUNDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class RunLaserRunDat : public IDataItem {
public:
  friend class EcalCondDBInterface;
  RunLaserRunDat();
  ~RunLaserRunDat() override;

  // User data methods
  inline std::string getTable() override { return "RUN_LASERRUN_CONFIG_DAT"; }

  inline void setLaserSequenceType(std::string x) { m_laserSeqType = x; }
  inline std::string getLaserSequenceType() const { return m_laserSeqType; }
  //
  inline void setLaserSequenceCond(std::string x) { m_laserSeqCond = x; }
  inline std::string getLaserSequenceCond() const { return m_laserSeqCond; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const RunLaserRunDat* item, RunIOV* iov) noexcept(false);

  void fetchData(std::map<EcalLogicID, RunLaserRunDat>* fillMap, RunIOV* iov) noexcept(false);

  // User data
  std::string m_laserSeqType;
  std::string m_laserSeqCond;
};

#endif
