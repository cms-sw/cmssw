#ifndef RUNPNERRORSDAT_H
#define RUNPNERRORSDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include <cstdint>

class RunPNErrorsDat : public IDataItem {
public:
  friend class EcalCondDBInterface;
  RunPNErrorsDat();
  ~RunPNErrorsDat() override;

  // User data methods
  inline std::string getTable() override { return "RUN_PN_ERRORS_DAT"; }

  inline void setErrorBits(uint64_t bits) { m_errorBits = bits; }
  inline uint64_t getErrorBits() const { return m_errorBits; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const RunPNErrorsDat* item, RunIOV* iov) noexcept(false);

  void fetchData(std::map<EcalLogicID, RunPNErrorsDat>* fillMap, RunIOV* iov) noexcept(false);

  // User data
  uint64_t m_errorBits;
};

#endif
