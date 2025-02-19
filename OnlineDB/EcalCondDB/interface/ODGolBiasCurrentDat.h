#ifndef ODGOLBIASCURRENTDAT_H
#define ODGOLBIASCURRENTDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/ODGolBiasCurrentInfo.h"

class ODGolBiasCurrentDat : public IODConfig {
 public:
  friend class EcalCondDBInterface;
  ODGolBiasCurrentDat();
  ~ODGolBiasCurrentDat();

  // User data methods
  inline std::string getTable() { return "GOL_BIAS_CURRENT_DAT"; }

  inline void setId(int dac) { m_ID = dac; }
  inline int getId() const { return m_ID; }

  inline void setGolId(int dac) { m_gol = dac; }
  inline int getGolId() const { return m_gol; }

  inline void setFedId(int dac) { m_fed = dac; }
  inline int getFedId() const { return m_fed; }

  inline void setTTId(int dac) { m_tt = dac; }
  inline int getTTId() const { return m_tt; }

  inline void setCurrent(int dac) { m_cur = dac; }
  inline int getCurrent() const { return m_cur; }

  inline void setPLLCurrent(int x) { m_pll_cur = x; }
  inline int getPLLCurrent() const { return m_pll_cur; }

  inline void setStatus(int dac) { m_sta = dac; }
  inline int getStatus() const { return m_sta; }


 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const ODGolBiasCurrentDat* item, ODGolBiasCurrentInfo* iov )
    throw(std::runtime_error);

  void writeArrayDB(const std::vector< ODGolBiasCurrentDat > data, ODGolBiasCurrentInfo* iov)
    throw(std::runtime_error);


  void fetchData(std::vector< ODGolBiasCurrentDat >* fillMap, ODGolBiasCurrentInfo* iov)
     throw(std::runtime_error);

  // User data
  int m_gol;
  int m_fed;
  int m_tt;
  int m_cur;
  int m_pll_cur;
  int m_sta;
  int m_ID;
 
};

#endif
