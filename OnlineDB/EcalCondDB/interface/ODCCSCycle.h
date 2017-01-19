#ifndef ODCCSCYCLE_H
#define ODCCSCYCLE_H

#include <stdexcept>
#include <iostream>

#include "OnlineDB/EcalCondDB/interface/IODConfig.h"



class ODCCSCycle :  public IODConfig  {
 public:
  friend class EcalCondDBInterface;
  friend class ODEcalCycle;

  ODCCSCycle();
  ~ODCCSCycle();

  inline std::string getTable() { return "ECAL_CCS_CYCLE"; }

  inline void setId(int id){m_ID=id;}
  inline int getId()const{ return m_ID;} ;

  // Methods for user data
  inline void setCCSConfigurationID(int x){m_ccs_config_id=x;}
  inline int getCCSConfigurationID() const{return m_ccs_config_id;}

  // Operators
  inline bool operator==(const ODCCSCycle &m) const { return ( m_ID   == m.m_ID); }
  inline bool operator!=(const ODCCSCycle &m) const { return !(*this == m); }



 private:
  // User data 
  int m_ID;
  int  m_ccs_config_id;
  void writeDB() noexcept(false);
  void prepareWrite() noexcept(false);
  void clear();
  void fetchData(ODCCSCycle * result) noexcept(false);

 void insertConfig() noexcept(false);

  // Methods from IUniqueDBObject
  int fetchID() noexcept(false);
  void setByID(int id) noexcept(false);


};

#endif
