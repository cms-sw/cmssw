#ifndef ODRUNCONFIGSEQINFO_H
#define ODRUNCONFIGSEQINFO_H

#include <stdexcept>
#include <iostream>

#include "OnlineDB/EcalCondDB/interface/IIOV.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/RunModeDef.h"
#include "OnlineDB/EcalCondDB/interface/RunTypeDef.h"
#include "OnlineDB/EcalCondDB/interface/RunSeqDef.h"

typedef int run_t;

class ODRunConfigSeqInfo : public IIOV {
 public:
  friend class EcalCondDBInterface;

  ODRunConfigSeqInfo();
  ~ODRunConfigSeqInfo();

  // Methods for user data
  void setID(int id) ; 
  int getID() ;
  void setDescription(std::string x);
  std::string getDescription() const;
  void setNumberOfCycles(int n);
  int getNumberOfCycles() const;
  void setSequenceNumber(int n);
  int getSequenceNumber() const;
  RunSeqDef getRunSeqDef() const;
  void setRunSeqDef(const RunSeqDef runSeqDef);

  // Methods from IUniqueDBObject
  int fetchID() throw(std::runtime_error); // fetches the sequence by the ecal_config_id and seq_num
  int fetchIDLast() throw(std::runtime_error); // fetches the sequence by the ecal_config_id and seq_num
  void setByID(int id) throw(std::runtime_error);

  // operators
  inline bool operator==(const ODRunConfigSeqInfo &r) const {  return (m_ID   == r.m_ID ); }
  inline bool operator!=(const ODRunConfigSeqInfo &r) const { return !(*this == r); }

 private:
  // User data for this IOV
  int m_ecal_config_id;
  int m_seq_num;
  int m_cycles;
  RunSeqDef m_run_seq;
  std::string m_description;

  int writeDB() throw(std::runtime_error);

};



#endif
