#ifndef ODRUNCONFIGINFO_H
#define ODRUNCONFIGINFO_H

#include <stdexcept>
#include <iostream>

#include "OnlineDB/EcalCondDB/interface/IIOV.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/RunModeDef.h"
#include "OnlineDB/EcalCondDB/interface/RunTypeDef.h"

typedef int run_t;

class ODRunConfigInfo : public IIOV {
 public:
  friend class EcalCondDBInterface;

  ODRunConfigInfo();
  ~ODRunConfigInfo();

  // Methods for user data
  Tm getDBTime() const;
  void setID(int id) ; 
  int getID() ;
  void setTag(std::string x);
  std::string getTag() const;
  void setDescription(std::string x);
  std::string getDescription() const;
  void setVersion(int vers);
  int getVersion() const;
  void setNumberOfSequences(int n);
  int getNumberOfSequences() const;
  RunTypeDef getRunTypeDef() const;
  void setRunTypeDef(const RunTypeDef runTypeDef);
  RunModeDef getRunModeDef() const;
  void setRunModeDef(const RunModeDef runModeDef);



  // Methods from IUniqueDBObject
  int fetchID() throw(std::runtime_error);
  int fetchIDFromTagAndVersion() throw(std::runtime_error);
  int fetchIDLast() throw(std::runtime_error);
  void setByID(int id) throw(std::runtime_error);

  // operators
  inline bool operator==(const ODRunConfigInfo &r) const {  return (m_ID   == r.m_ID ); }
  inline bool operator!=(const ODRunConfigInfo &r) const { return !(*this == r); }

 private:
  // User data for this IOV
  Tm m_db_time;
  std::string m_tag;
  int m_version;
  RunModeDef m_runModeDef;
  RunTypeDef m_runTypeDef;
  int m_num_seq;
  std::string m_description;

  int writeDB() throw(std::runtime_error);

};



#endif
