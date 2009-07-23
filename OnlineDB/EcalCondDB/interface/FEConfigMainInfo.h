#ifndef FECONFIGMAIN_H
#define FECONFIGMAIN_H

#include <map>
#include <string>

#include "OnlineDB/EcalCondDB/interface/IODConfig.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"

class FEConfigMainInfo : public  IODConfig {
 public:
  friend class EcalCondDBInterface;

  FEConfigMainInfo();
  ~FEConfigMainInfo();

  // Methods for user data
  inline std::string getTable() { return "FE_CONFIG_MAIN"; }


  inline void setId(int id) { m_ID = id; }
  inline int getId() const { return m_ID; }
  // for compatibility
  void setID(int id) {setId(id);} 
  int getID() { return getId() ;}
  // the tag is already in IODConfig 


  inline void setPedId(int id) { m_ped = id; }
  inline int getPedId() const { return m_ped ;}
  inline void setLinId(int id) { m_lin= id; }
  inline int getLinId() const { return m_lin; }
  inline void setLutId(int id) { m_lut= id; }
  inline int getLutId() const { return m_lut; }
  inline void setFgrId(int id) { m_fgr = id; }
  inline int getFgrId() const { return m_fgr; }
  inline void setSliId(int id) { m_sli = id; }
  inline int getSliId() const { return m_sli; }
  inline void setWeiId(int id) { m_wei = id; }
  inline int getWeiId() const { return m_wei; }
  inline void setBxtId(int id) { m_bxt = id; }
  inline int getBxtId() const { return m_bxt; }
  inline void setBttId(int id) { m_btt = id; }
  inline int getBttId() const { return m_btt; }


  Tm getDBTime() const{  return m_db_time;}
  void setDBTime(Tm x) { m_db_time=x; } 


  // Methods from IUniqueDBObject
  int fetchID() throw(std::runtime_error);
  int fetchNextId() throw(std::runtime_error);
  void setByID(int id) throw(std::runtime_error);
  void setParameters(std::map<string,string> my_keys_map);

  // operators
  inline bool operator==(const FEConfigMainInfo &r) const {  return (m_ID   == r.m_ID ); }
  inline bool operator!=(const FEConfigMainInfo &r) const { return !(*this == r); }

 private:
  // User data for this IOV

  int m_ID;
  Tm m_db_time;

  int m_ped;
  int m_lin;
  int m_lut;
  int m_fgr;
  int m_sli;
  int m_wei;
  int m_bxt;
  int m_btt;



  void prepareWrite()  throw(std::runtime_error);
  void writeDB() throw(std::runtime_error);
  void clear();
  void fetchData(FEConfigMainInfo * result)     throw(std::runtime_error);
  void fetchLastData(FEConfigMainInfo * result)     throw(std::runtime_error);


};


#endif
