#ifndef LMFPNIREDDAT_H
#define LMFPNIREDDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunTag.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class LMFPNIRedDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  LMFPNIRedDat();
  ~LMFPNIRedDat();

  // User data methods
  inline std::string getTable() { return "LMF_PN_IRED_DAT"; }

  inline void setPNPeak(float peak) { m_pnPeak = peak; }
  inline float getPNPeak() const { return m_pnPeak; }

  inline void setPNErr(float err) { m_pnErr = err; }
  inline float getPNErr() const { return m_pnErr; }

  
 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const LMFPNIRedDat* item, LMFRunIOV* iov)
    throw(std::runtime_error);

  void fetchData(std::map< EcalLogicID, LMFPNIRedDat >* fillVec, LMFRunIOV* iov)
     throw(std::runtime_error);

  // User data
  float m_pnPeak;
  float m_pnErr;
  
};

#endif
