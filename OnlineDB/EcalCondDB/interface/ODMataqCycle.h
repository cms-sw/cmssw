#ifndef ODMATAQCYCLE_H
#define ODMATAQCYCLE_H

#include <stdexcept>
#include <iostream>

#include "OnlineDB/EcalCondDB/interface/IODConfig.h"



class ODMataqCycle :  public IODConfig  {
 public:
  friend class EcalCondDBInterface;
  friend class ODEcalCycle;

  ODMataqCycle();
  ~ODMataqCycle();

  inline std::string getTable() { return "ECAL_Mataq_CYCLE"; }

  inline void setId(int id){m_ID=id;}
  inline int getId()const{ return m_ID;} ;

  // Methods for user data
  inline void setMataqConfigurationID(int x){m_mataq_config_id=x;}
  inline int getMataqConfigurationID() const{return m_mataq_config_id;}

  // Operators
  inline bool operator==(const ODMataqCycle &m) const { return ( m_ID   == m.m_ID); }
  inline bool operator!=(const ODMataqCycle &m) const { return !(*this == m); }

 private:
  // User data 
  int m_ID;
  int  m_mataq_config_id;
  void writeDB() throw(std::runtime_error);
  void prepareWrite()  throw(std::runtime_error);
  void clear();
  void fetchData(ODMataqCycle * result)     throw(std::runtime_error);
 void insertConfig() throw(std::runtime_error);


  // Methods from IUniqueDBObject
  int fetchID() throw(std::runtime_error);
  void setByID(int id) throw(std::runtime_error);


};

#endif
