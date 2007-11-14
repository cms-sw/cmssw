#ifndef FECONFIGWEIGHTS_H
#define FECONFIGWEIGHTS_H

#include <stdexcept>
#include <iostream>

#include "OnlineDB/EcalCondDB/interface/IIOV.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"


typedef int run_t;

class FEConfigWeightsInfo : public IIOV {
 public:
  friend class EcalCondDBInterface;

  FEConfigWeightsInfo();
  ~FEConfigWeightsInfo();

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
  void setByID(int id) throw(std::runtime_error);

  // operators
  inline bool operator==(const FEConfigWeightsInfo &r) const {  return (m_ID   == r.m_ID ); }
  inline bool operator!=(const FEConfigWeightsInfo &r) const { return !(*this == r); }

 private:
  // User data for this IOV
  int m_ngr ;
  Tm m_db_time;
  std::string m_tag;

  int writeDB() throw(std::runtime_error);

};



#endif
