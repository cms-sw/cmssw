#ifndef CALIHVSCANRATIODAT_H
#define CALIHVSCANRATIODAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/CaliTag.h"
#include "OnlineDB/EcalCondDB/interface/CaliIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class CaliHVScanRatioDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  CaliHVScanRatioDat();
  ~CaliHVScanRatioDat();
  
  // User data methods
  inline std::string getTable() { return "CALI_HV_SCAN_RATIO_DAT"; }

  inline void setHVRatio(float c) { m_hvratio = c; }
  inline float getHVRatio() const { return m_hvratio; }

  inline void setHVRatioRMS(float c) { m_hvratioRMS = c; }
  inline float getHVRatioRMS() const { return m_hvratioRMS; }

  inline void setTaskStatus(bool s) { m_taskStatus = s; }
  inline bool getTaskStatus() const { return m_taskStatus; }

 private:
  void prepareWrite() 
    throw(std::runtime_error);
  
  void writeDB(const EcalLogicID* ecid, const CaliHVScanRatioDat* item, CaliIOV* iov)
    throw(std::runtime_error);
  
  void fetchData(std::map< EcalLogicID, CaliHVScanRatioDat >* fillVec, CaliIOV* iov)
    throw(std::runtime_error);
  
  // User data
  float m_hvratio;
  float m_hvratioRMS;
  bool m_taskStatus;
  
  
};

#endif
