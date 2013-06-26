#ifndef OBADXTDAT_H
#define OBADXTDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IODConfig.h"
#include "OnlineDB/EcalCondDB/interface/ODBadXTInfo.h"

class ODBadXTDat : public IODConfig {
 public:
  friend class EcalCondDBInterface;
  ODBadXTDat();
  ~ODBadXTDat();

  // User data methods
  inline std::string getTable() { return "BAD_CRYSTALS_DAT"; }

  inline void setId(int dac) { m_ID = dac; }
  inline int getId() const { return m_ID; }

  inline void setSMId(int dac) { m_sm = dac; }
  inline int getSMId() const { return m_sm; }

  inline void setFedId(int dac) { m_fed = dac; }
  inline int getFedId() const { return m_fed; }

  inline void setTTId(int dac) { m_tt = dac; }
  inline int getTTId() const { return m_tt; }

  inline void setXTId(int dac) { m_xt = dac; }
  inline int getXTId() const { return m_xt; }


  inline void setStatus(int dac) { m_t1 = dac; }
  inline int getStatus() const { return m_t1; }

 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const ODBadXTDat* item, ODBadXTInfo* iov )
    throw(std::runtime_error);

  void writeArrayDB(const std::vector< ODBadXTDat > data, ODBadXTInfo* iov)
    throw(std::runtime_error);


  void fetchData(std::vector< ODBadXTDat >* fillMap, ODBadXTInfo* iov)
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
