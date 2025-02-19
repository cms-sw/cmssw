#ifndef ODTOWERSTOBYPASSDAT_H
#define ODTOWERSTOBYPASSDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/ODTowersToByPassInfo.h"

class ODTowersToByPassDat : public IODConfig {
 public:
  friend class EcalCondDBInterface;
  ODTowersToByPassDat();
  ~ODTowersToByPassDat();

  // User data methods
  inline std::string getTable() { return "TOWERS_TO_BYPASS_DAT"; }

  inline void setId(int dac) { m_ID = dac; }
  inline int getId() const { return m_ID; }

  inline void setTrId(int dac) { m_tr = dac; }
  inline int getTrId() const { return m_tr; }

  inline void setFedId(int dac) { m_fed = dac; }
  inline int getFedId() const { return m_fed; }

  inline void setTTId(int dac) { m_tt = dac; }
  inline int getTTId() const { return m_tt; }

  inline void setTimeCorr(int dac) { m_time = dac; }
  inline int getTimeCorr() const { return m_time; }

  inline void setStatus(int dac) { m_sta = dac; }
  inline int getStatus() const { return m_sta; }


 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const ODTowersToByPassDat* item, ODTowersToByPassInfo* iov )
    throw(std::runtime_error);

  void writeArrayDB(const std::vector< ODTowersToByPassDat > data, ODTowersToByPassInfo* iov)
    throw(std::runtime_error);


  void fetchData(std::vector< ODTowersToByPassDat >* fillMap, ODTowersToByPassInfo* iov)
     throw(std::runtime_error);

  // User data
  int m_tr;
  int m_fed;
  int m_tt;
  int m_time; 
  int m_sta;
  int m_ID;
 
};

#endif
