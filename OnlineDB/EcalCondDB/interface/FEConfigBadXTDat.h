#ifndef FECONFIGBADXTDAT_H
#define FECONFIGBADXTDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IODConfig.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigBadXTInfo.h"

class FEConfigBadXTDat : public IODConfig {
 public:
  friend class EcalCondDBInterface;
  FEConfigBadXTDat();
  ~FEConfigBadXTDat();

  // User data methods
  inline std::string getTable() { return "FE_CONFIG_BadCRYSTALS_DAT"; }

  inline void setId(int dac) { m_ID = dac; }
  inline int getId() const { return m_ID; }

  inline void setTRId(int dac) { m_sm = dac; }
  inline int getTRId() const { return m_sm; }

  inline void setTCCId(int dac) { m_fed = dac; }
  inline int getTCCId() const { return m_fed; }

  inline void setTTId(int dac) { m_tt = dac; }
  inline int getTTId() const { return m_tt; }

  inline void setXTId(int dac) { m_xt = dac; }
  inline int getXTId() const { return m_xt; }


  inline void setStatus(int dac) { m_t1 = dac; }
  inline int getStatus() const { return m_t1; }

 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const FEConfigBadXTDat* item, FEConfigBadXTInfo* iov )
    throw(std::runtime_error);

  void writeArrayDB(const std::vector< FEConfigBadXTDat > data, FEConfigBadXTInfo* iov)
    throw(std::runtime_error);


  void fetchData(std::vector< FEConfigBadXTDat >* fillMap, FEConfigBadXTInfo* iov)
     throw(std::runtime_error);

  // User data
  int m_sm;
  int m_fed;
  int m_tt;
  int m_xt;
  int m_t1;
  int m_ID;
 
};

#endif
