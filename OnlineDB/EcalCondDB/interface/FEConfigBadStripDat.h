#ifndef FECONFIGBADStripDAT_H
#define FECONFIGBADStripDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IODConfig.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigBadStripInfo.h"

class FEConfigBadStripDat : public IODConfig {
 public:
  friend class EcalCondDBInterface;
  FEConfigBadStripDat();
  ~FEConfigBadStripDat();

  // User data methods
  inline std::string getTable() { return "FE_CONFIG_BadCRYSTALS_DAT"; }

  inline void setId(int dac) { m_ID = dac; }
  inline int getId() const { return m_ID; }

  inline void setFedId(int dac) { m_fed = dac; }
  inline int getFedId() const { return m_fed; }

  inline void setTCCId(int dac) { m_tcc = dac; }
  inline int getTCCId() const { return m_tcc; }

  inline void setTTId(int dac) { m_tt = dac; }
  inline int getTTId() const { return m_tt; }

  inline void setStripId(int dac) { m_xt = dac; }
  inline int getStripId() const { return m_xt; }


  inline void setStatus(int dac) { m_t1 = dac; }
  inline int getStatus() const { return m_t1; }

 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const FEConfigBadStripDat* item, FEConfigBadStripInfo* iov )
    throw(std::runtime_error);

  void writeArrayDB(const std::vector< FEConfigBadStripDat > data, FEConfigBadStripInfo* iov)
    throw(std::runtime_error);


  void fetchData(std::vector< FEConfigBadStripDat >* fillMap, FEConfigBadStripInfo* iov)
     throw(std::runtime_error);

  // User data
  int m_tcc;
  int m_fed;
  int m_tt;
  int m_xt;
  int m_t1;
  int m_ID;
 
};

#endif
