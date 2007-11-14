#ifndef FECONFPARAMDAT_H
#define FECONFPARAMDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigLinInfo.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class FEConfigParamDat : public IDataItem {
 public:
  friend class EcalCondDBInterface; // XXX temp should not need
  FEConfigParamDat();
  ~FEConfigParamDat();

  // User data methods
  inline std::string getTable() { return "FE_CONFIG_PARAM_DAT"; }


  inline void setETSat(float x) { m_etsat = x; }
  inline void setTTThreshlow(float x) { m_tthreshlow = x; }
  inline void setTTThreshhigh(float x) { m_tthreshhigh = x; }
  inline void setFGlowthresh(float x) { m_fglowthresh = x; }
  inline void setFGhighthresh(float x) { m_fghighthresh = x; }
  inline void setFGlowratio(float x) { m_lowratio = x; }
  inline void setFGhighratio(float x) { m_highratio = x; }

  inline float getETSat() const       { return m_etsat; }
  inline float getTTThreshlow() const { return m_tthreshlow; }
  inline float getTTThreshhigh() const{ return m_tthreshhigh; }
  inline float getFGlowthresh() const { return m_fglowthresh; }
  inline float getFGhighthresh() const{ return m_fghighthresh; }
  inline float getFGlowratio() const  { return m_lowratio ; }
  inline float getFGhighratio() const { return m_highratio; }

 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const FEConfigParamDat* item, FEConfigLinInfo* iconf)
    throw(std::runtime_error);


  void writeArrayDB(const std::map< EcalLogicID, FEConfigParamDat>* data, FEConfigLinInfo* iconf)
  throw(std::runtime_error);


  void fetchData(std::map< EcalLogicID, FEConfigParamDat >* fillMap, FEConfigLinInfo* iconf)
     throw(std::runtime_error);

  // User data
float  m_etsat ;
float  m_tthreshlow ;
float  m_tthreshhigh ;
float  m_fglowthresh ;
float  m_fghighthresh ;
float  m_lowratio ;
float  m_highratio ;

};

#endif
