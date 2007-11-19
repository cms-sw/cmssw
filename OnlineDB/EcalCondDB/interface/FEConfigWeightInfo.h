#ifndef FECONFIGWEIGHT_H
#define FECONFIGWEIGHT_H

#include <stdexcept>
#include <iostream>

#include "OnlineDB/EcalCondDB/interface/IIOV.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"


typedef int run_t;

class FEConfigWeightInfo : public IIOV {
 public:
  friend class EcalCondDBInterface;

  FEConfigWeightInfo();
  ~FEConfigWeightInfo();

  // Methods for user data
  Tm getDBTime() const;
  void setNumberOfGroups(int n);
  int getNumberOfGroups() const;
  void setID(int id) ; 
  int getID() ;
  void setTag(std::string x);
  std::string getTag() const;

  // Methods from IUniqueDBObject
  int fetchID() throw(std::runtime_error);
  int fetchIDFromTag() throw(std::runtime_error);
  int fetchIDLast() throw(std::runtime_error);
  void setByID(int id) throw(std::runtime_error);

  // operators
  inline bool operator==(const FEConfigWeightInfo &r) const {  return (m_ID   == r.m_ID ); }
  inline bool operator!=(const FEConfigWeightInfo &r) const { return !(*this == r); }

 private:
  // User data for this IOV
  int m_ngr ;
  Tm m_db_time;
  std::string m_tag;

  int writeDB() throw(std::runtime_error);

};



#endif
