#ifndef RUNTTERRORSDAT_H
#define RUNTTERRORSDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include <cstdint>

class RunTTErrorsDat : public IDataItem {
public:
  friend class EcalCondDBInterface;
  RunTTErrorsDat();
  ~RunTTErrorsDat() override;

  // User data methods
  inline std::string getTable() override { return "RUN_TT_ERRORS_DAT"; }

  inline void setErrorBits(uint64_t bits) { m_errorBits = bits; }
  inline uint64_t getErrorBits() const { return m_errorBits; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const RunTTErrorsDat* item, RunIOV* iov) noexcept(false);

  void fetchData(std::map<EcalLogicID, RunTTErrorsDat>* fillMap, RunIOV* iov) noexcept(false);

  // User data
  uint64_t m_errorBits;
};

#endif
