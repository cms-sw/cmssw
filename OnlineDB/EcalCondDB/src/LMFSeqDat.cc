#include <stdexcept>
#include <sstream>
#include <limits.h>
#include "OnlineDB/EcalCondDB/interface/LMFSeqDat.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"

using namespace std;
using namespace oracle::occi;

LMFSeqDat::LMFSeqDat() : LMFUnique()
{
  setClassName("LMFSeqDat");

  m_runIOV = RunIOV();
  m_intFields["seq_num"] = 0;
  Tm t;
  t = t.plusInfinity();
  m_stringFields["seq_start"] = t.str();
  m_stringFields["seq_stop"]  = t.str();
  m_intFields["vmin"] = 1;
  m_intFields["vmax"] = 0;
}

LMFSeqDat::LMFSeqDat(EcalDBConnection *c) : LMFUnique(c) {
  setClassName("LMFSeqDat");

  m_runIOV = RunIOV();
  m_intFields["seq_num"] = 0;
  Tm t;
  t = t.plusInfinity();
  m_stringFields["seq_start"] = t.str();
  m_stringFields["seq_stop"]  = t.str();
  m_intFields["vmin"] = 1;
  m_intFields["vmax"] = 0;
}

LMFSeqDat::LMFSeqDat(oracle::occi::Environment* env,
		     oracle::occi::Connection* conn) : LMFUnique(env, conn) {
  setClassName("LMFSeqDat");

  m_runIOV = RunIOV();
  m_intFields["seq_num"] = 0;
  Tm t;
  t = t.plusInfinity();
  m_stringFields["seq_start"] = t.str();
  m_stringFields["seq_stop"]  = t.str();
  m_intFields["vmin"] = 1;
  m_intFields["vmax"] = 0;
}

LMFSeqDat::~LMFSeqDat()
{
}

Tm LMFSeqDat::getSequenceStop() const {
  Tm t;
  t.setToString(getString("seq_stop"));
  return t;
}

RunIOV LMFSeqDat::getRunIOV() const 
{ 
  return m_runIOV;
}

LMFSeqDat& LMFSeqDat::setRunIOV(const RunIOV &iov)
{
  if (iov != m_runIOV) {
    m_ID = 0;
    m_runIOV = iov;
  }
  return *this;
}

std::string LMFSeqDat::fetchIdSql(Statement *stmt)
{
  int runIOVID = m_runIOV.getID();
  if (!runIOVID) { 
    if (m_debug) {
      std::cout << m_className << ": RunIOV not set" << endl;
    }
    return "";
  }

  if (m_debug) {
    std::cout << "Run IOV ID: " << runIOVID << std::endl;
    std::cout << "SEQ #     : " << getSequenceNumber() << std::endl;
    std::cout << "Versions  : " << getVmin() << " - " << getVmax() << endl;
  }
  std::string sql = "SELECT SEQ_ID FROM LMF_SEQ_DAT "
    "WHERE "
    "RUN_IOV_ID   = :1 AND "
    "SEQ_NUM      = :2 AND "
    "VMIN         = :3 ";
  if (getVmax() > 0) {
    sql += "AND VMAX = :4";
  } else {
    sql += "ORDER BY VMAX DESC";
  }
  stmt->setSQL(sql);
  stmt->setInt(1, runIOVID);
  stmt->setInt(2, getSequenceNumber());
  stmt->setInt(3, getVmin());
  if (getVmax() > 0) {
    stmt->setInt(4, getVmax());
  }
  return sql;
}

std::string LMFSeqDat::setByIDSql(Statement *stmt, int id) {
  std::string sql = "SELECT RUN_IOV_ID, SEQ_NUM, SEQ_START, SEQ_STOP, " 
    "VMIN, VMAX FROM LMF_SEQ_DAT WHERE SEQ_ID = :1";
  stmt->setSQL(sql);
  stmt->setInt(1, id);
  return sql;
}

void LMFSeqDat::getParameters(ResultSet *rset) {     
  DateHandler dh(m_env, m_conn);
  int runIOVID = rset->getInt(1);
  setInt("seq_num", rset->getInt(2));
  Date startDate = rset->getDate(3);
  Date endDate = rset->getDate(4);
  setInt("vmin", rset->getInt(5));
  setInt("vmax", rset->getInt(6));

  setString("seq_start", dh.dateToTm( startDate ).str());
  setString("seq_stop",  dh.dateToTm( endDate ).str());

  m_runIOV.setConnection(m_env, m_conn);
  m_runIOV.setByID(runIOVID);
}

bool LMFSeqDat::isValid() const {
  bool ret = true;
  if (getSequenceStart().isNull()) {
    ret = false;
  }
  if ((getSequenceStop().str().length() > 0) &&
      (getSequenceStop().microsTime() < getSequenceStart().microsTime())) {
    ret = false;
  }
  if (getSequenceStop() == Tm().plusInfinity()) {
    ret = false;
  }
  return ret;
}

std::string LMFSeqDat::writeDBSql(Statement *stmt)
{
  // Validate the data, use infinity-till convention
  DateHandler dh(m_env, m_conn);

  if (!isValid()) {
    dump();
    throw(std::runtime_error("LMFSeqDat::writeDB: not valid"));
  }

  if (getSequenceStop().str().length() == 0) {
    setSequenceStop(dh.getPlusInfTm());
  }
  int runIOVID = m_runIOV.getID();
  if (runIOVID == 0) {
    throw(std::runtime_error("LMFSeqDat::writeDB: RunIOV not set"));
  }
  std::string sp = sequencePostfix(getSequenceStart());
  std::string sql = "INSERT INTO LMF_SEQ_DAT (SEQ_ID, RUN_IOV_ID, SEQ_NUM, "
    "SEQ_START, SEQ_STOP, VMIN, VMAX) "
    "VALUES (SEQ_ID_" + sp + "_SQ.NextVal, :1, :2, :3, :4, :5, :6)";
  cout << sql << endl;
  stmt->setSQL(sql);
  stmt->setInt(1, runIOVID);
  stmt->setInt(2, getSequenceNumber());
  stmt->setDate(3, dh.tmToDate(getSequenceStart()));
  stmt->setDate(4, dh.tmToDate(getSequenceStop()));
  stmt->setInt(5, getVmin());
  stmt->setInt(6, getVmax());
  return sql;
}

void LMFSeqDat::fetchParentIDs()
  throw(std::runtime_error)
{
  // get the RunIOV
  m_runIOV.setConnection(m_env, m_conn);
  int runIOVID = m_runIOV.getID();
  m_runIOV.setByID(runIOVID);

  if (m_runIOV.getID() == 0) { 
    throw(std::runtime_error("LMFSeqDat:  Given RunIOV does not exist in DB")); 
  }

}

std::map<int, LMFSeqDat> LMFSeqDat::fetchByRunIOV(std::string sql,
						  std::string method)
  throw(std::runtime_error)
{
  std::vector<std::string> pars;
  return fetchByRunIOV(pars, sql, method);
}

std::map<int, LMFSeqDat> LMFSeqDat::fetchByRunIOV(int par, 
						  std::string sql,
						  std::string method)
  throw(std::runtime_error)
{
  std::vector<std::string> pars;
  std::stringstream ss;
  ss << "I" << par;
  pars.push_back(ss.str());
  return fetchByRunIOV(pars, sql, method);
}

std::map<int, LMFSeqDat> LMFSeqDat::fetchByRunIOV(std::vector<std::string> pars, 
						  std::string sql,
						  std::string method)
  throw(std::runtime_error)
{
  std::map<int, LMFSeqDat> l;
  this->checkConnection();
  try {
    Statement *stmt = m_conn->createStatement();
    stmt->setSQL(sql);
    for (unsigned int i = 0; i < pars.size(); i++) {
      if (pars[i][0] == 'I') {
	stmt->setInt(i + 1, atoi(pars[i].c_str() + 1));
      } else if (pars[i][0] == 'S') {
	stmt->setString(i + 1, pars[i].c_str() + 1);
      } else {
	throw(std::runtime_error(m_className + "::" + method + ": " + 
				 "Invalid type"));
      }
    }
    ResultSet *rset = stmt->executeQuery();
    while (rset->next()) {
      int seq_id = rset->getInt(1);
      LMFSeqDat s;
      s.setConnection(m_env, m_conn);
      s.setByID(seq_id);
      l[s.getSequenceNumber()] = s;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error(m_className + "::" + method + ": " + 
			     e.getMessage()));
  }
  return l;
}

LMFSeqDat LMFSeqDat::fetchLast() {
  LMFSeqDat ret;
  std::map<int, LMFSeqDat> m = 
    fetchByRunIOV("SELECT SEQ_ID FROM LMF_SEQ_DAT WHERE SEQ_ID = "
		  "(SELECT MAX(SEQ_ID) FROM LMF_SEQ_DAT)", "fetchLast");
  if (m.size() > 0) {
    ret = m.begin()->second;
  }
  return ret;
}

RunIOV LMFSeqDat::fetchLastRun() {
  return fetchLast().getRunIOV();
}

std::map<int, LMFSeqDat> LMFSeqDat::fetchByRunIOV(RunIOV &iov) {
  int runIOVID = iov.getID();
  return fetchByRunIOV(runIOVID, 
		       "SELECT SEQ_ID FROM LMF_SEQ_DAT WHERE RUN_IOV_ID = :1",
		       "fetchByRunIOV");
}

std::map<int, LMFSeqDat> LMFSeqDat::fetchByRunIOV(RunIOV &iov, 
						  const LMFColor &col) {
  int runIOVID = iov.getID();
  int colorId  = col.getID();
  std::vector<std::string> pars;
  std::stringstream ss;
  ss << "I" << runIOVID;
  pars.push_back(ss.str());
  ss.str(std::string());
  ss << "I" << colorId;
  pars.push_back(ss.str());
  return fetchByRunIOV(pars, 
		       "SELECT S.SEQ_ID FROM LMF_SEQ_DAT S JOIN LMF_RUN_IOV R"
		       " ON S.SEQ_ID = R.SEQ_ID WHERE RUN_IOV_ID = :1 AND "
		       " COLOR_ID = :2",
		       "fetchByRunIOVAndColor");
}

std::map<int, LMFSeqDat> LMFSeqDat::fetchByRunNumber(int runno) {
  return fetchByRunIOV(runno, 
		       "SELECT SEQ_ID FROM LMF_SEQ_DAT D JOIN RUN_IOV R ON "
		       "D.RUN_IOV_ID = R.IOV_ID WHERE RUN_NUM = :1",
		       "fetchByRunNumber");
}

LMFSeqDat LMFSeqDat::fetchByRunNumber(int runno, Tm taken_at) {
  return fetchByRunNumber(runno, taken_at.str());
}

LMFSeqDat LMFSeqDat::fetchByRunNumber(int runno, std::string taken_at) {
  std::map<int, LMFSeqDat> l;
  std::vector<std::string> pars;
  std::stringstream ss;
  ss << "I" << runno;
  pars.push_back(ss.str());
  ss.str(std::string());
  ss << "S" << taken_at;
  pars.push_back(ss.str());
  std::string q = "SELECT SEQ_ID FROM LMF_SEQ_DAT D JOIN RUN_IOV R ON "
    "D.RUN_IOV_ID = R.IOV_ID WHERE RUN_NUM = :1 AND "
    "SEQ_START >= TO_DATE(:2, 'YYYY-MM-DD HH24:MI:SS') "
    "AND SEQ_STOP <= TO_DATE(:2, 'YYYY-MM-DD HH24:MI:SS')";
  l = fetchByRunIOV(pars, q, "fetchByRunNumberAt");
  LMFSeqDat ret;
  if (l.size() == 1) {
    std::map<int, LMFSeqDat>::const_iterator x = l.begin();
    ret = x->second;
  } else if (l.size() > 1) {
    std::cout << "WARNING: Your query returned more than one result. " 
	      << std::endl;
    std::cout << "         This was not expected. Please check the DB!!!" 
	      << std::endl;
    std::cout << "Your query: " << std::endl << q << std::endl;
    std::cout << "Your parameters: " << std::endl;
    for (unsigned int i = 0; i < pars.size(); i++) {
      std::cout << i << ": " << pars[i] << std::endl;
    }
  }
  return ret;
}

