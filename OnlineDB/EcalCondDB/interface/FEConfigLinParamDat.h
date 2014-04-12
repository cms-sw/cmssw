#ifndef FECONFLINPARAMDAT_H
#define FECONFLINPARAMDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigLinInfo.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class FEConfigLinParamDat : public IDataItem {
 public:
  friend class EcalCondDBInterface; 
  FEConfigLinParamDat();
  ~FEConfigLinParamDat();

  // User data methods
  inline std::string getTable() { return "FE_CONFIG_LINPARAM_DAT"; }


  inline void setETSat(float x) { m_etsat = x; }

  inline float getETSat() const       { return m_etsat; }

 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const FEConfigLinParamDat* item, FEConfigLinInfo* iconf)
    throw(std::runtime_error);


  void writeArrayDB(const std::map< EcalLogicID, FEConfigLinParamDat>* data, FEConfigLinInfo* iconf)
  throw(std::runtime_error);


  void fetchData(std::map< EcalLogicID, FEConfigLinParamDat >* fillMap, FEConfigLinInfo* iconf)
     throw(std::runtime_error);

  // User data
float  m_etsat ;

};

#endif
