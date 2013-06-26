#ifndef RUNCRYSTALERRORSDAT_H
#define RUNCRYSTALERRORSDAT_H

#include <vector>
#include <stdexcept>
#include <boost/cstdint.hpp>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class RunCrystalErrorsDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  RunCrystalErrorsDat();
  ~RunCrystalErrorsDat();

  // User data methods
  inline std::string getTable() { return "RUN_CRYSTAL_ERRORS_DAT"; }

  inline void setErrorBits(uint64_t bits) { m_errorBits = bits; }
  inline uint64_t getErrorBits() const { return m_errorBits; }

 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const RunCrystalErrorsDat* item, RunIOV* iov )
    throw(std::runtime_error);

  void fetchData(std::map< EcalLogicID, RunCrystalErrorsDat >* fillMap, RunIOV* iov)
     throw(std::runtime_error);

  // User data
  uint64_t m_errorBits;
};

#endif
