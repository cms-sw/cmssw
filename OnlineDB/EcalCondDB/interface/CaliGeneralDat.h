#ifndef CALIGENERALDAT_H
#define CALIGENERALDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/CaliTag.h"
#include "OnlineDB/EcalCondDB/interface/CaliIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class CaliGeneralDat : public IDataItem {
public:
  friend class EcalCondDBInterface;
  CaliGeneralDat();
  ~CaliGeneralDat() override;

  // User data methods
  inline std::string getTable() override { return "CALI_GENERAL_DAT"; }

  inline void setNumEvents(int n) { m_numEvents = n; }
  inline int getNumEvents() const { return m_numEvents; }

  inline void setComments(std::string comments) { m_comments = comments; }
  inline std::string getComments() const { return m_comments; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const CaliGeneralDat* item, CaliIOV* iov) noexcept(false);

  void fetchData(std::map<EcalLogicID, CaliGeneralDat>* fillVec, CaliIOV* iov) noexcept(false);

  // User data
  int m_numEvents;
  std::string m_comments;
};

#endif
