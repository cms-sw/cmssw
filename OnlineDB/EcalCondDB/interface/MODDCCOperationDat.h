#ifndef MODDCCOPERATIONDAT_H
#define MODDCCOPERATIONDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/MODRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class MODDCCOperationDat : public IDataItem {
public:
  friend class EcalCondDBInterface;
  MODDCCOperationDat();
  ~MODDCCOperationDat() override;

  // User data methods
  inline std::string getTable() override { return "OD_DCC_OPERATION_DAT"; }

  inline void setOperation(std::string x) { m_word = x; }
  inline std::string getOperation() const { return m_word; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const MODDCCOperationDat* item, MODRunIOV* iov) noexcept(false);

  void writeArrayDB(const std::map<EcalLogicID, MODDCCOperationDat>* data, MODRunIOV* iov) noexcept(false);

  void fetchData(std::map<EcalLogicID, MODDCCOperationDat>* fillMap, MODRunIOV* iov) noexcept(false);

  // User data
  std::string m_word;
};

#endif
