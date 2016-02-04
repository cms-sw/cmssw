#ifndef FECONFLUTPARAMDAT_H
#define FECONFLUTPARAMDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigLUTInfo.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class FEConfigLUTParamDat : public IDataItem {
 public:
  friend class EcalCondDBInterface; // XXX temp should not need
  FEConfigLUTParamDat();
  ~FEConfigLUTParamDat();

  // User data methods
  inline std::string getTable() { return "FE_CONFIG_LUTPARAM_DAT"; }


  inline void setETSat(float x) { m_etsat = x; }
  inline void setTTThreshlow(float x) { m_tthreshlow = x; }
  inline void setTTThreshhigh(float x) { m_tthreshhigh = x; }



  inline float getETSat() const       { return m_etsat; }
  inline float getTTThreshlow() const { return m_tthreshlow; }
  inline float getTTThreshhigh() const{ return m_tthreshhigh; }

 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const FEConfigLUTParamDat* item, FEConfigLUTInfo* iconf)
    throw(std::runtime_error);


  void writeArrayDB(const std::map< EcalLogicID, FEConfigLUTParamDat>* data, FEConfigLUTInfo* iconf)
  throw(std::runtime_error);


  void fetchData(std::map< EcalLogicID, FEConfigLUTParamDat >* fillMap, FEConfigLUTInfo* iconf)
     throw(std::runtime_error);

  // User data
float  m_etsat ;
float  m_tthreshlow ;
float  m_tthreshhigh ;

};

#endif
