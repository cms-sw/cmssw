#ifndef FECONFIGWEIGHT_H
#define FECONFIGWEIGHT_H

#include <map>
#include <string>

#include "OnlineDB/EcalCondDB/interface/IODConfig.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"

class FEConfigWeightInfo : public  IODConfig {
 public:
  friend class EcalCondDBInterface;

  FEConfigWeightInfo();
  ~FEConfigWeightInfo();

  // Methods for user data
  inline std::string getTable() { return "FE_CONFIG_WEIGHT_INFO"; }


  void setNumberOfGroups(int n){m_ngr=n;}
  int getNumberOfGroups() const{return m_ngr;}

  inline void setId(int id) { m_ID = id; }
  inline int getId() const { return m_ID; }
  // for compatibility
  void setID(int id) {setId(id);} 
  int getID() { return getId() ;}
  // the tag is already in IODConfig 
  inline void setVersion(int id) { m_version = id; }
  inline int getVersion() const { return m_version; }


  Tm getDBTime() const{  return m_db_time;}
  void setDBTime(Tm x) { m_db_time=x; } 


  // Methods from IUniqueDBObject
  int fetchID() throw(std::runtime_error);
  int fetchNextId() throw(std::runtime_error);
  void setByID(int id) throw(std::runtime_error);
  void setParameters(std::map<std::string,std::string> my_keys_map);

  // operators
  inline bool operator==(const FEConfigWeightInfo &r) const {  return (m_ID   == r.m_ID ); }
  inline bool operator!=(const FEConfigWeightInfo &r) const { return !(*this == r); }

 private:
  // User data for this IOV
  int m_ngr ;
  int m_ID;
  Tm m_db_time;
  int m_version;

  void prepareWrite()  throw(std::runtime_error);
  void writeDB() throw(std::runtime_error);
  void clear();
  void fetchData(FEConfigWeightInfo * result)     throw(std::runtime_error);
  void fetchLastData(FEConfigWeightInfo * result)     throw(std::runtime_error);


};


#endif
