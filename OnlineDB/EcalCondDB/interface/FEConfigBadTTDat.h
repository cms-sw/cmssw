#ifndef FECONFIGBADTTDAT_H
#define FECONFIGBADTTDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IODConfig.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigBadTTInfo.h"

class FEConfigBadTTDat : public IODConfig {
 public:
  friend class EcalCondDBInterface;
  FEConfigBadTTDat();
  ~FEConfigBadTTDat();

  // User data methods
  inline std::string getTable() { return "FE_CONFIG_BadTT_DAT"; }

  inline void setId(int dac) { m_ID = dac; }
  inline int getId() const { return m_ID; }

  inline void setFedId(int x) { m_fed = x; }
  inline int getFedId() const { return m_fed; }

  inline void setTCCId(int dac) { m_tcc = dac; }
  inline int getTCCId() const { return m_tcc; }

  inline void setTTId(int dac) { m_tt = dac; }
  inline int getTTId() const { return m_tt; }


  inline void setStatus(int dac) { m_t1 = dac; }
  inline int getStatus() const { return m_t1; }

 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const FEConfigBadTTDat* item, FEConfigBadTTInfo* iov )
    throw(std::runtime_error);

  void writeArrayDB(const std::vector< FEConfigBadTTDat > data, FEConfigBadTTInfo* iov)
    throw(std::runtime_error);


  void fetchData(std::vector< FEConfigBadTTDat >* fillMap, FEConfigBadTTInfo* iov)
     throw(std::runtime_error);

  // User data
  int m_tcc;
  int m_fed;
  int m_tt;
  int m_t1;
  int m_ID;
 
};

#endif
