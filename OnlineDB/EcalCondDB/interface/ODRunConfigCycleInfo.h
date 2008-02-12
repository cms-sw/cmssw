#ifndef ODRUNCONFIGCYCLEINFO_H
#define ODRUNCONFIGCYCLEINFO_H

#include <stdexcept>
#include <iostream>

#include "OnlineDB/EcalCondDB/interface/IIOV.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/RunModeDef.h"
#include "OnlineDB/EcalCondDB/interface/RunTypeDef.h"
#include "OnlineDB/EcalCondDB/interface/RunSeqDef.h"

typedef int run_t;

class ODRunConfigCycleInfo : public IIOV {
 public:
  friend class EcalCondDBInterface;

  ODRunConfigCycleInfo();
  ~ODRunConfigCycleInfo();

  // Methods for user data
  void setID(int id) ; 
  int getID() ;
  void setTag(std::string x);
  std::string getTag() const;
  void setDescription(std::string x);
  std::string getDescription() const;
  void setCycleNumber(int n);
  int getCycleNumber() const;

  void setSequenceID(int n);
  int getSequenceID() const;

  // Methods from IUniqueDBObject
  int fetchID() throw(std::runtime_error); // fetches the Cycle by the seq_id and cycle_num
  int fetchIDLast() throw(std::runtime_error); // fetches the Cycle by the seq_id and cycle_num
  void setByID(int id) throw(std::runtime_error);

  // operators
  inline bool operator==(const ODRunConfigCycleInfo &r) const {  return (m_ID   == r.m_ID ); }
  inline bool operator!=(const ODRunConfigCycleInfo &r) const { return !(*this == r); }

 private:
  // User data for this IOV
  int m_sequence_id;
  int m_cycle_num;
  std::string m_tag;
  std::string m_description;

  int writeDB() throw(std::runtime_error);

};



#endif
