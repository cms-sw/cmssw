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
  ~MonShapeQualityDat();

  // User data methods
  inline std::string getTable() { return "MON_SHAPE_QUALITY_DAT"; }

  inline void setAvgChi2(float chi2) { m_avgChi2 = chi2; }
  inline float getAvgChi2() const { return m_avgChi2; }
  
 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const MonShapeQualityDat* item, MonRunIOV* iov)
    throw(std::runtime_error);

  void writeArrayDB(const std::map< EcalLogicID, MonShapeQualityDat >* data, MonRunIOV* iov)
    throw(std::runtime_error);

  void fetchData(std::map< EcalLogicID, MonShapeQualityDat >* fillVec, MonRunIOV* iov)
     throw(std::runtime_error);

  // User data
  float m_avgChi2;
  
};

#endif
