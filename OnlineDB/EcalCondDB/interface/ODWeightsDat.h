#ifndef ODWEIGHTSDAT_H
#define ODWEIGHTSDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/IODConfig.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/ODFEWeightsInfo.h"

class ODWeightsDat : public IODConfig {
 public:
  friend class EcalCondDBInterface;
  ODWeightsDat();
  ~ODWeightsDat();

  // User data methods
  inline std::string getTable() { return "DCC_WEIGHTS_DAT"; }

  inline void setId(int dac) { m_ID = dac; }
  inline int getId() const { return m_ID; }

  inline void setSMId(int dac) { m_sm = dac; }
  inline int getSMId() const { return m_sm; }

  inline void setFedId(int dac) { m_fed = dac; }
  inline int getFedId() const { return m_fed; }

  inline void setTTId(int dac) { m_tt = dac; }
  inline int getTTId() const { return m_tt; }

  inline void setCrystalId(int dac) { m_xt = dac; }
  inline int getCrystalId() const { return m_xt; }

  inline void setWeight0( float x) { m_wei0 = x; }
  inline void setWeight1( float x) { m_wei1 = x; }
  inline void setWeight2( float x) { m_wei2 = x; }
  inline void setWeight3( float x) { m_wei3 = x; }
  inline void setWeight4( float x) { m_wei4 = x; }
  inline void setWeight5( float x) { m_wei5 = x; }

  inline float getWeight0() const { return m_wei0; }
  inline float getWeight1() const { return m_wei1; }
  inline float getWeight2() const { return m_wei2; }
  inline float getWeight3() const { return m_wei3; }
  inline float getWeight4() const { return m_wei4; }
  inline float getWeight5() const { return m_wei5; }


 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const ODWeightsDat* item, ODFEWeightsInfo* iov )
    throw(std::runtime_error);

  void writeArrayDB(const std::vector< ODWeightsDat > data, ODFEWeightsInfo* iov)
    throw(std::runtime_error);


  void fetchData(std::vector< ODWeightsDat >* fillMap, ODFEWeightsInfo* iov)
     throw(std::runtime_error);

  // User data
  int m_sm;
  int m_fed;
  int m_tt;
  int m_xt;
  int m_ID;
  float m_wei0;
  float m_wei1;
  float m_wei2;
  float m_wei3;
  float m_wei4;
  float m_wei5;
 
};

#endif
