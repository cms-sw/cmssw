#ifndef ODRUNCONFIGSEQINFO_H
#define ODRUNCONFIGSEQINFO_H

#include <stdexcept>
#include <iostream>

#include "OnlineDB/EcalCondDB/interface/IODConfig.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/RunModeDef.h"
#include "OnlineDB/EcalCondDB/interface/RunTypeDef.h"
#include "OnlineDB/EcalCondDB/interface/RunSeqDef.h"

typedef int run_t;

class ODRunConfigSeqInfo : public IODConfig {
 public:
  friend class EcalCondDBInterface;

  ODRunConfigSeqInfo();
  ~ODRunConfigSeqInfo();

  inline std::string getTable() { return "ECAL_SEQUENCE_DAT"; }


  // Methods for user data

  inline void setId(int id) { m_ID = id; }
  inline int getId() const { return m_ID; }

  inline   void setDescription(std::string x) { m_description = x; }
  inline  std::string getDescription() const{ return m_description;}
  inline  void setEcalConfigId(int x){ m_ecal_config_id = x; }
  inline  int getEcalConfigId()const{ return m_ecal_config_id;}
  inline  void setNumberOfCycles(int x){ m_cycles = x; }
  inline  void setSequenceId(int x){ m_ID = x; }
  inline  int getSequenceId()const{ return m_ID;}
  inline  int getNumberOfCycles() const{return m_cycles;}
  inline  void setSequenceNumber(int x){m_seq_num=x;}
  inline  int getSequenceNumber() const{return m_seq_num;}
  //
  RunSeqDef getRunSeqDef() const;
  void setRunSeqDef(const RunSeqDef runSeqDef);

  // operators
  inline bool operator==(const ODRunConfigSeqInfo &r) const {  return (m_ID   == r.m_ID ); }
  inline bool operator!=(const ODRunConfigSeqInfo &r) const { return !(*this == r); }

 private:
  int m_ID;
  int m_ecal_config_id;
  int m_seq_num;
  int m_sequence_id;
  int m_cycles;
  RunSeqDef m_run_seq;
  std::string m_description;

  // Methods from IUniqueDBObject
  int fetchID() throw(std::runtime_error); // fetches the sequence by the ecal_config_id and seq_num
  int fetchIDLast() throw(std::runtime_error); // fetches the sequence by the ecal_config_id and seq_num
  void setByID(int id) throw(std::runtime_error);

  void  writeDB()throw(std::runtime_error);

  void prepareWrite()  throw(std::runtime_error);

  void fetchData(ODRunConfigSeqInfo * result)     throw(std::runtime_error);
  void clear();


};



#endif
