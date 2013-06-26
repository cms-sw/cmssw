#ifndef OBADTTDAT_H
#define OBADTTDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IODConfig.h"
#include "OnlineDB/EcalCondDB/interface/ODBadTTInfo.h"

class ODBadTTDat : public IODConfig {
 public:
  friend class EcalCondDBInterface;
  ODBadTTDat();
  ~ODBadTTDat();

  // User data methods
  inline std::string getTable() { return "BAD_TT_DAT"; }

  inline void setId(int dac) { m_ID = dac; }
  inline int getId() const { return m_ID; }

  inline void setSMId(int dac) { m_sm = dac; }
  inline int getSMId() const { return m_sm; }

  inline void setTRId(int dac) { m_sm = dac; }
  inline int getTRId() const { return m_sm; }

  inline void setFedId(int dac) { m_fed = dac; }
  inline int getFedId() const { return m_fed; }

  inline void setTTId(int dac) { m_tt = dac; }
  inline int getTTId() const { return m_tt; }


  inline void setStatus(int dac) { m_t1 = dac; }
  inline int getStatus() const { return m_t1; }

 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const ODBadTTDat* item, ODBadTTInfo* iov )
    throw(std::runtime_error);

  void writeArrayDB(const std::vector< ODBadTTDat > data, ODBadTTInfo* iov)
    throw(std::runtime_error);


  void fetchData(std::vector< ODBadTTDat >* fillMap, ODBadTTInfo* iov)
     throw(std::runtime_error);

  // User data
  int m_sm;
  int m_fed;
  int m_tt;
  int m_t1;
  int m_ID;
 
};

#endif
