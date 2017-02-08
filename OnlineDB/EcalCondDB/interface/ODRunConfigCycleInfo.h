#ifndef ODRUNCONFIGCYCLEINFO_H
#define ODRUNCONFIGCYCLEINFO_H

#include <stdexcept>
#include <iostream>


#include "OnlineDB/EcalCondDB/interface/RunModeDef.h"
#include "OnlineDB/EcalCondDB/interface/RunTypeDef.h"
#include "OnlineDB/EcalCondDB/interface/RunSeqDef.h"
#include "OnlineDB/EcalCondDB/interface/IODConfig.h"


class ODRunConfigCycleInfo : public IODConfig {
 public:
  friend class EcalCondDBInterface;
  friend class ODEcalCycle;

  ODRunConfigCycleInfo();
  ~ODRunConfigCycleInfo();

  inline std::string getTable() { return "ECAL_CYCLE_DAT"; }

  // Methods for user data
  inline void setId(int id) { m_ID = id; }
  inline int getId() const { return m_ID; }



void setDescription(std::string x) { m_description = x;}
std::string getDescription() const{  return m_description;}
//
void setTag(std::string x) { m_tag = x;}
std::string getTag() const{  return m_tag;}
//
void setSequenceID(int x) { m_sequence_id = x;}
int getSequenceID() const{  return m_sequence_id;}
//
void setCycleNumber(int n){ m_cycle_num = n;  }
int getCycleNumber()const {return m_cycle_num;  }
//



  // operators
  inline bool operator==(const ODRunConfigCycleInfo &r) const {  return (m_ID   == r.m_ID ); }
  inline bool operator!=(const ODRunConfigCycleInfo &r) const { return !(*this == r); }

 private:
  // User data for this IOV
  int m_ID;
  int m_sequence_id;
  int m_cycle_num;
  std::string m_tag;
  std::string m_description;

  void prepareWrite()  throw(std::runtime_error);
  void writeDB()       throw(std::runtime_error);
  void clear();
  void fetchData(ODRunConfigCycleInfo * result)     throw(std::runtime_error);
  void insertConfig() throw(std::runtime_error);


  // Methods from IUniqueDBObject
  int fetchID() throw(std::runtime_error); // fetches the Cycle by the seq_id and cycle_num
  int fetchIDLast() throw(std::runtime_error); // fetches the Cycle by the seq_id and cycle_num
  void setByID(int id) throw(std::runtime_error);

};



#endif
