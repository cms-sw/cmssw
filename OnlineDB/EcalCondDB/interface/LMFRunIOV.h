#ifndef LMFRUNIOV_H
#define LMFRUNIOV_H

#include <stdexcept>
#include <iostream>

#include "OnlineDB/EcalCondDB/interface/IIOV.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"

typedef int subrun_t;

class LMFRunIOV : public IIOV {
 public:
  friend class EcalCondDBInterface;

  LMFRunIOV();
  ~LMFRunIOV();

  // Methods for user data
  void setLMFRunTag(LMFRunTag tag);
  LMFRunTag getLMFRunTag() const;
  void setRunIOV(RunIOV iov);
  RunIOV getRunIOV();

  inline void setSubRunNumber(subrun_t subrun){m_subRunNum=subrun;}
  inline run_t getSubRunNumber() const{return m_subRunNum;}

  inline int getSequenceNumber() const{return m_subRunNum/1000000;}
  inline int getLMRNumber() const{return (m_subRunNum-(m_subRunNum/1000000)*1000000)/100;}
  inline int getLaserType() const{return (m_subRunNum%100);}
  inline void setSubRunStart(Tm start){m_subRunStart=start;}
  inline Tm getSubRunStart() const{return m_subRunStart;}
  inline void setSubRunEnd(Tm end){m_subRunEnd=end;}
  inline Tm getSubRunEnd() const{return m_subRunEnd;}
  inline void setDBInsertionTime(Tm x){m_dbtime=x;}
  inline Tm getDBInsertionTime() const{return m_dbtime;}
  inline void setSubRunType(std::string x){m_subrun_type=x;}
  inline std::string getSubRunType() const{return m_subrun_type;}

  void setID(int id);

  // Methods from IUniqueDBObject
  int getID(){ return m_ID;} ;
  int fetchID() throw(std::runtime_error);
  void setByID(int id) throw(std::runtime_error);

  // Operators
  inline bool operator==(const LMFRunIOV &m) const
    {
      return ( m_lmfRunTag   == m.m_lmfRunTag &&
	       m_runIOV      == m.m_runIOV &&
	       m_subRunNum   == m.m_subRunNum &&
	       m_subRunStart == m.m_subRunStart &&
	       m_subRunEnd   == m.m_subRunEnd );
    }

  inline bool operator!=(const LMFRunIOV &m) const { return !(*this == m); }

 private:
  // User data for this IOV
  LMFRunTag m_lmfRunTag;
  RunIOV m_runIOV;
  subrun_t m_subRunNum;
  Tm m_subRunStart;
  Tm m_subRunEnd;
  Tm m_dbtime;
  std::string m_subrun_type;

  int writeDB() throw(std::runtime_error);
  void fetchParentIDs(int* lmfRunTagID, int* runIOVID) throw(std::runtime_error);

  void setByRun(LMFRunTag* lmftag, RunIOV* runiov, subrun_t subrun) throw(std::runtime_error);
};

#endif
