#ifndef FECONFIGLUT_H
#define FECONFIGLUT_H

#include <map>
#include <string>

#include "OnlineDB/EcalCondDB/interface/IODConfig.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"

class FEConfigLUTInfo : public  IODConfig {
 public:
  friend class EcalCondDBInterface;

  FEConfigLUTInfo();
  ~FEConfigLUTInfo();

  // Methods for user data
  inline std::string getTable() { return "FE_CONFIG_LUT_INFO"; }


  inline void setNumberOfGroups(int iov_id){ m_iov_id = iov_id;  }
  int getNumberOfGroups() const{return m_iov_id;  }

  inline void setId(int id) { m_ID = id; }
  inline int getId() const { return m_ID; }
  // for compatibility
  void setID(int id) {setId(id);} 
  int getID() { return getId() ;}
  // the tag is already in IODConfig 
  inline void setVersion(int id) { m_version = id; }
  inline int getVersion() const { return m_version; }


  Tm getDBTime() const{  return m_db_time;}
  void setDBTime(const Tm& x) { m_db_time=x; } 


  // Methods from IUniqueDBObject
  int fetchID() noexcept(false);
  int fetchNextId() noexcept(false);
  void setByID(int id) noexcept(false);
  void setParameters(const std::map<std::string,std::string>& my_keys_map);

  // operators
  inline bool operator==(const FEConfigLUTInfo &r) const {  return (m_ID   == r.m_ID ); }
  inline bool operator!=(const FEConfigLUTInfo &r) const { return !(*this == r); }

 private:
  // User data for this IOV
  int m_iov_id ;
  int m_ID;
  Tm m_db_time;
  int m_version;

  void prepareWrite() noexcept(false);
  void writeDB() noexcept(false);
  void clear();
  void fetchData(FEConfigLUTInfo * result) noexcept(false);
  void fetchLastData(FEConfigLUTInfo * result) noexcept(false);


};


#endif
