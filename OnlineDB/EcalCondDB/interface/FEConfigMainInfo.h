#ifndef FECONFIGMAININFO_H
#define FECONFIGMAININFO_H

#include <stdexcept>
#include <iostream>


#include <map>
#include <string>


#include "OnlineDB/EcalCondDB/interface/IODConfig.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"




class FEConfigMainInfo : public IODConfig {
 public:
  friend class EcalCondDBInterface;

  FEConfigMainInfo();
  ~FEConfigMainInfo();

  inline std::string getTable() { return "FE_CONFIG_MAIN"; }

  // Methods for user data
  inline void setId(int id) { m_ID = id; }
  inline int getId() const { return m_ID; }

  Tm getDBTime() const{  return m_db_time;}
  void setDBTime(Tm x) { m_db_time=x; } 


void setDescription(std::string x) { m_description = x;}
std::string getDescription() const{  return m_description;}
//
void setPedId(int x) { m_ped_id = x;}
int getPedId() const{  return m_ped_id;}
//
void setLinId(int x){ m_lin_id = x;  }
int getLinId()const {return m_lin_id;  }
//
void setLUTId(int x){ m_lut_id = x;  }
int getLUTId()const {return m_lut_id;  }
//
void setFgrId(int x) { m_fgr_id = x;}
int getFgrId() const{  return m_fgr_id;}
//
void setSliId(int x){ m_sli_id = x;  }
int getSliId()const {return m_sli_id;  }
//
void setWeiId(int x) { m_wei_id = x;}
int getWeiId() const{  return m_wei_id;}
//
void setSpiId(int x) { m_spi_id = x;}
int getSpiId() const{  return m_spi_id;}
//
void setTimId(int x) { m_tim_id = x;}
int getTimId() const{  return m_tim_id;}
//
void setBxtId(int x){ m_bxt_id = x;  }
int getBxtId()const {return m_bxt_id;  }
//
void setBttId(int x){ m_btt_id = x;  }
int getBttId()const {return m_btt_id;  }
//
void setBstId(int x){ m_bst_id = x;  }
int getBstId()const {return m_bst_id;  }
//
 inline void setVersion(int id) { m_version = id; }
 inline int getVersion() const { return m_version; }



  // operators
  inline bool operator==(const FEConfigMainInfo &r) const {  return (m_ID   == r.m_ID ); }
  inline bool operator!=(const FEConfigMainInfo &r) const { return !(*this == r); }

 private:
  // User data for this IOV
  int m_ID;
  int m_ped_id;
  int m_lin_id;
  int m_lut_id;
  int m_sli_id;
  int m_fgr_id;
  int m_wei_id;
  int m_bxt_id;
  int m_btt_id;
  int m_bst_id;
  int m_tim_id;
  int m_spi_id;
  int m_version;
  Tm m_db_time;
  std::string m_description;

  void prepareWrite()  throw(std::runtime_error);
  void writeDB()       throw(std::runtime_error);
  void clear();
  void fetchData(FEConfigMainInfo * result)     throw(std::runtime_error);
  void insertConfig() throw(std::runtime_error);


  // Methods from IUniqueDBObject
  int fetchID() throw(std::runtime_error); // fetches 
  int fetchNextId() throw(std::runtime_error); // fetches 
  int fetchIDLast() throw(std::runtime_error); // fetches the last one
  void setByID(int id) throw(std::runtime_error);

};



#endif
