#ifndef RUNCOMMENTDAT_H
#define RUNCOMMENTDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class RunCommentDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  RunCommentDat();
  ~RunCommentDat();

  // User data methods
  inline std::string getTable() { return "RUN_COMMENT_DAT"; }

  inline void setSource(std::string x) { m_source = x; }
  inline std::string getSource() const { return m_source; }
  inline void setComment(std::string x) { m_comment = x; }
  inline std::string getComment() const { return m_comment; }
  inline void setDBTime(Tm x) {m_time=x;}
  inline Tm getDBTime() const {return m_time;}

 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const RunCommentDat* item, RunIOV* iov )
    throw(std::runtime_error);

  void fetchData(std::map< EcalLogicID, RunCommentDat >* fillMap, RunIOV* iov)
     throw(std::runtime_error);

  // User data
 std::string m_source ;
 std::string m_comment ;
 Tm  m_time ;
};

#endif
