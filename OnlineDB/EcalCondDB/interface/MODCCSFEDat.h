#ifndef MODCCSFEDAT_H
#define MODCCSFEDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/MODRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class MODCCSFEDat : public IDataItem {
public:
  friend class EcalCondDBInterface;
  MODCCSFEDat();
  ~MODCCSFEDat() override;

  // User data methods
  inline std::string getTable() override { return "OD_CCS_FE_DAT"; }

  inline void setWord(int x) { m_word = x; }
  inline int getWord() const { return m_word; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const MODCCSFEDat* item, MODRunIOV* iov) noexcept(false);

  void writeArrayDB(const std::map<EcalLogicID, MODCCSFEDat>* data, MODRunIOV* iov) noexcept(false);

  void fetchData(std::map<EcalLogicID, MODCCSFEDat>* fillMap, MODRunIOV* iov) noexcept(false);

  // User data
  int m_word;
};

#endif
