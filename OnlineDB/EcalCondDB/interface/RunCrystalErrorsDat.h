#ifndef RUNCRYSTALERRORSDAT_H
#define RUNCRYSTALERRORSDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include <cstdint>

class RunCrystalErrorsDat : public IDataItem {
public:
  friend class EcalCondDBInterface;
  RunCrystalErrorsDat();
  ~RunCrystalErrorsDat() override;

  // User data methods
  inline std::string getTable() override { return "RUN_CRYSTAL_ERRORS_DAT"; }

  inline void setErrorBits(uint64_t bits) { m_errorBits = bits; }
  inline uint64_t getErrorBits() const { return m_errorBits; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const RunCrystalErrorsDat* item, RunIOV* iov) noexcept(false);

  void fetchData(std::map<EcalLogicID, RunCrystalErrorsDat>* fillMap, RunIOV* iov) noexcept(false);

  // User data
  uint64_t m_errorBits;
};

#endif
