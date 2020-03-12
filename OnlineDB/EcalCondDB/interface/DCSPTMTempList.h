#ifndef DCSPTMTEMPLIST_H
#define DCSPTMTEMPLIST_H

#include <stdexcept>
#include <iostream>

#include "OnlineDB/EcalCondDB/interface/IIOV.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"
#include "OnlineDB/EcalCondDB/interface/DCSPTMTemp.h"
#include "OnlineDB/EcalCondDB/interface/DCSPTMTempList.h"

typedef int run_t;

class DCSPTMTempList : public IDBObject {
public:
  friend class EcalCondDBInterface;

  DCSPTMTempList();
  ~DCSPTMTempList() override;

  // Methods for user data

  std::vector<DCSPTMTemp> getList();

  // this fills the vector
  void fetchValuesForECIDAndTime(const EcalLogicID& ecid, const Tm& start, const Tm& end) noexcept(false);
  void fetchValuesForECID(const EcalLogicID& ecid) noexcept(false);

private:
  // User data for this IOV
  std::vector<DCSPTMTemp> m_vec_temp;
};

#endif
