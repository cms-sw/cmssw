#ifndef FECONFIGLIN_H
#define FECONFIGLIN_H

#include <map>
#include <string>

#include "OnlineDB/EcalCondDB/interface/IODConfig.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"

class FEConfigLinInfo : public  IODConfig {
 public:
  friend class EcalCondDBInterface;

  FEConfigLinInfo();
  ~FEConfigLinInfo();

  // Methods for user data
  inline std::string getTable() { return "FE_CONFIG_LIN_INFO"; }


  inline void setIOVId(int iov_id){ m_iov_id = iov_id;  }
  int getIOVId() const{return m_iov_id;  }

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
  inline bool operator==(const FEConfigLinInfo &r) const {  return (m_ID   == r.m_ID ); }
  inline bool operator!=(const FEConfigLinInfo &r) const { return !(*this == r); }

 private:
  // User data for this IOV
  int m_iov_id ;
  int m_ID;
  Tm m_db_time;
  int m_version;

  void prepareWrite()  throw(std::runtime_error);
  void writeDB() throw(std::runtime_error);
  void clear();
  void fetchData(FEConfigLinInfo * result)     throw(std::runtime_error);
  void fetchLastData(FEConfigLinInfo * result)     throw(std::runtime_error);


};


#endif
