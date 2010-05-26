#ifndef LMFPNTESTPULSEDAT_H
#define LMFPNTESTPULSEDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunTag.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class LMFPNTestPulseDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  LMFPNTestPulseDat();
  ~LMFPNTestPulseDat();

  // User data methods
  inline std::string getTable() { return "LMF_PN_TEST_PULSE_DAT"; }

  inline void setADCMean(float mean) { m_adcMean = mean; }
  inline float getADCMean() const { return m_adcMean; }

  inline void setADCRMS(float rms) { m_adcRMS = rms; }
  inline float getADCRMS() const { return m_adcRMS; }

  
 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const LMFPNTestPulseDat* item, LMFRunIOV* iov)
    throw(std::runtime_error);

  void fetchData(std::map< EcalLogicID, LMFPNTestPulseDat >* fillVec, LMFRunIOV* iov)
     throw(std::runtime_error);

  // User data
  float m_adcMean;
  float m_adcRMS;
  
};

#endif
