#ifndef LMFPNBLUEDAT_H
#define LMFPNBLUEDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunTag.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class LMFPNBlueDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  LMFPNBlueDat();
  ~LMFPNBlueDat();

  // User data methods
  inline std::string getTable() { return "LMF_PN_BLUE_DAT"; }

  inline void setPNPeak(float peak) { m_pnPeak = peak; }
  inline float getPNPeak() const { return m_pnPeak; }

  inline void setPNErr(float err) { m_pnErr = err; }
  inline float getPNErr() const { return m_pnErr; }

  
 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const LMFPNBlueDat* item, LMFRunIOV* iov)
    throw(std::runtime_error);
  
  void writeArrayDB(const std::map< EcalLogicID, LMFPNBlueDat >* data, LMFRunIOV* iov)
     throw(runtime_error);

  void fetchData(std::map< EcalLogicID, LMFPNBlueDat >* fillVec, LMFRunIOV* iov)
     throw(std::runtime_error);

  // User data
  float m_pnPeak;
  float m_pnErr;
  
};

#endif
