#ifndef ODJBH4CYCLE_H
#define ODJBH4CYCLE_H

#include <stdexcept>
#include <iostream>

#include "OnlineDB/EcalCondDB/interface/IODConfig.h"



class ODJBH4Cycle :  public IODConfig  {
 public:
  friend class EcalCondDBInterface;
  friend class ODEcalCycle;

  ODJBH4Cycle();
  ~ODJBH4Cycle();

  inline std::string getTable() { return "ECAL_JBH4_CYCLE"; }

  inline void setId(int id){m_ID=id;}
  inline int getId()const{ return m_ID;} ;

  // Methods for user data
  inline void setJBH4ConfigurationID(int x){m_jbh4_config_id=x;}
  inline int getJBH4ConfigurationID() const{return m_jbh4_config_id;}

  // Operators
  inline bool operator==(const ODJBH4Cycle &m) const { return ( m_ID   == m.m_ID); }
  inline bool operator!=(const ODJBH4Cycle &m) const { return !(*this == m); }

 private:
  // User data 
  int m_ID;
  int  m_jbh4_config_id;
  void writeDB() throw(std::runtime_error);
  void prepareWrite()  throw(std::runtime_error);
  void clear();
  void fetchData(ODJBH4Cycle * result)     throw(std::runtime_error);
   void insertConfig() throw(std::runtime_error);


  // Methods from IUniqueDBObject
  int fetchID() throw(std::runtime_error);
  void setByID(int id) throw(std::runtime_error);


};

#endif
