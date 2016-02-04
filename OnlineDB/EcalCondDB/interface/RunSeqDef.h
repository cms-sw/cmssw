#ifndef RUNSEQDEF_H
#define RUNSEQDEF_H

#include <string>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDef.h"
#include "OnlineDB/EcalCondDB/interface/RunTypeDef.h"

/**
 *   Def for Location information
 */
class RunSeqDef : public IDef {
  public:
 friend class ODRunConfigSeqInfo; 
 friend class EcalCondDBInterface;
  
  RunSeqDef();
  virtual ~RunSeqDef();

  // Methods for user data
  std::string getRunSeq() const;
  void setRunSeq(std::string runseq);

  RunTypeDef getRunTypeDef() const;
  void setRunTypeDef(const RunTypeDef runTypeDef);




  
  // Methods from IUniqueDBObject
  int fetchID() throw(std::runtime_error);
  void setByID(int id) throw(std::runtime_error);

  // Operators.  m_desc is not considered, it cannot be written to DB anyhow
  inline bool operator==(const RunSeqDef &t) const { return m_runSeq == t.m_runSeq; }
  inline bool operator!=(const RunSeqDef &t) const { return m_runSeq != t.m_runSeq; }
  
 protected:
  // User data for this def
  std::string m_runSeq;
  RunTypeDef m_runType;  

  int writeDB()throw(std::runtime_error);


  void fetchAllDefs( std::vector<RunSeqDef>* fillVec) throw(std::runtime_error);
};

#endif
