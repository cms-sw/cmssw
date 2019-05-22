#ifndef MONSHAPEQUALITYDAT_H
#define MONSHAPEQUALITYDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/MonRunTag.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class MonShapeQualityDat : public IDataItem {
public:
  friend class EcalCondDBInterface;
  MonShapeQualityDat();
  ~MonShapeQualityDat() override;

  // User data methods
  inline std::string getTable() override { return "MON_SHAPE_QUALITY_DAT"; }

  inline void setAvgChi2(float chi2) { m_avgChi2 = chi2; }
  inline float getAvgChi2() const { return m_avgChi2; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const MonShapeQualityDat* item, MonRunIOV* iov) noexcept(false);

  void writeArrayDB(const std::map<EcalLogicID, MonShapeQualityDat>* data, MonRunIOV* iov) noexcept(false);

  void fetchData(std::map<EcalLogicID, MonShapeQualityDat>* fillVec, MonRunIOV* iov) noexcept(false);

  // User data
  float m_avgChi2;
};

#endif
