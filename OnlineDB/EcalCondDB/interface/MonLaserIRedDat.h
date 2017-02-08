#ifndef MONLASERIREDDAT_H
#define MONLASERIREDDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/MonRunTag.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class MonLaserIRedDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  MonLaserIRedDat();
  ~MonLaserIRedDat();

  // User data methods
  inline std::string getTable() { return "MON_LASER_IRED_DAT"; }

  inline void setAPDMean(float mean) { m_apdMean = mean; }
  inline float getAPDMean() const { return m_apdMean; }
  
  inline void setAPDRMS(float rms) { m_apdRMS = rms; }
  inline float getAPDRMS() const { return m_apdRMS; }

  inline void setAPDOverPNMean(float mean) { m_apdOverPNMean = mean; }
  inline float getAPDOverPNMean() const { return m_apdOverPNMean; }
  
  inline void setAPDOverPNRMS(float rms) { m_apdOverPNRMS = rms; }
  inline float getAPDOverPNRMS() const { return m_apdOverPNRMS; }
  
  inline void setTaskStatus(bool status) { m_taskStatus = status; }
  inline bool getTaskStatus() const { return m_taskStatus; }


 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const MonLaserIRedDat* item, MonRunIOV* iov)
    throw(std::runtime_error);

  void writeArrayDB(const std::map< EcalLogicID, MonLaserIRedDat >* data, MonRunIOV* iov)
    throw(std::runtime_error);

  void fetchData(std::map< EcalLogicID, MonLaserIRedDat >* fillMap, MonRunIOV* iov)
     throw(std::runtime_error);

  // User data
  float m_apdMean;
  float m_apdRMS;
  float m_apdOverPNMean;
  float m_apdOverPNRMS;
  bool m_taskStatus;
  
};

#endif
