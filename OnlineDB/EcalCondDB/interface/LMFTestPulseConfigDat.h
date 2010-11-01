#ifndef LMFTPCFGDAT_H
#define LMFTPCFGDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunTag.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class LMFTestPulseConfigDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  LMFTestPulseConfigDat();
  ~LMFTestPulseConfigDat();

  // User data methods
  inline std::string getTable() { return "LMF_TEST_PULSE_CONFIG_DAT"; }

  inline void setVFEGain(int x) { m_vfe_gain = x; }
  inline int getVFEGain() const { return m_vfe_gain; }

  inline void setDACMGPA(int x) { m_dac_mgpa = x; }
  inline int getDACMGPA() const { return m_dac_mgpa; }

  inline void setPNGain(int x) { m_pn_gain = x; }
  inline int getPNGain() const { return m_pn_gain; }

  inline void setPNVinj(int x) { m_pn_vinj = x; }
  inline int getPNVinj() const { return m_pn_vinj; }
  
 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const LMFTestPulseConfigDat* item, LMFRunIOV* iov)
    throw(std::runtime_error);
  
  void writeArrayDB(const std::map< EcalLogicID, LMFTestPulseConfigDat >* data, LMFRunIOV* iov)
     throw(runtime_error);

  void fetchData(std::map< EcalLogicID, LMFTestPulseConfigDat >* fillVec, LMFRunIOV* iov)
     throw(std::runtime_error);

  // User data
  int m_vfe_gain;
  int m_dac_mgpa;
  int m_pn_gain;
  int m_pn_vinj;

};

#endif
