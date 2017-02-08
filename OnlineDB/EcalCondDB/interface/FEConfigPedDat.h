#ifndef FECONFPEDDAT_H
#define FECONFPEDDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigPedInfo.h"

class FEConfigPedDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  FEConfigPedDat();
  ~FEConfigPedDat();

  // User data methods
  inline std::string getTable() { return "FE_CONFIG_PED_DAT"; }

  inline void setId(int x) { m_ID = x; }
  inline int getId() const { return m_ID; }

  inline void setPedMeanG1(float mean) { m_pedMeanG1 = mean; }
  inline float getPedMeanG1() const { return m_pedMeanG1; }

  inline void setPedMeanG6(float mean) { m_pedMeanG6 = mean; }
  inline float getPedMeanG6() const { return m_pedMeanG6; }

  inline void setPedMeanG12(float mean) { m_pedMeanG12 = mean; }
  inline float getPedMeanG12() const { return m_pedMeanG12; }


 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const FEConfigPedDat* item, FEConfigPedInfo* iconf )
    throw(std::runtime_error);

  void writeArrayDB(const std::map< EcalLogicID, FEConfigPedDat >* data, FEConfigPedInfo* iconf)
    throw(std::runtime_error);

  void fetchData(std::map< EcalLogicID, FEConfigPedDat >* fillMap, FEConfigPedInfo* iconf)
     throw(std::runtime_error);

  // User data
  float m_pedMeanG1;
  float m_pedMeanG6;
  float m_pedMeanG12;
  int m_ID;
 
};

#endif


