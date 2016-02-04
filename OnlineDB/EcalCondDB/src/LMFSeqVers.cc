#include "OnlineDB/EcalCondDB/interface/LMFSeqVers.h"

using namespace std;
using namespace oracle::occi;

LMFSeqVers::LMFSeqVers() : LMFPrimVers()
{
  setClassName("LMFSeqVers");
  setString("description", "");
}

LMFSeqVers::LMFSeqVers(EcalDBConnection *c) : LMFPrimVers(c) {
  setClassName("LMFSeqVers");
  setString("description", "");
}

LMFSeqVers::LMFSeqVers(oracle::occi::Environment* env,
		       oracle::occi::Connection* conn) : LMFPrimVers(env, conn) {
  setClassName("LMFSeqVers");
  setString("description", "");
}

LMFSeqVers::~LMFSeqVers()
{
}

std::string LMFSeqVers::fetchIdSql(Statement *stmt) {
  return "";
}

std::string LMFSeqVers::setByIDSql(Statement *stmt, int id) 
{
  std::string sql = "SELECT DESCR FROM LMF_SEQ_VERS "
    "WHERE VERS = :1";
  stmt->setSQL(sql);
  stmt->setInt(1, id);
  return sql;
}

void LMFSeqVers::getParameters(ResultSet *rset) {
  setString("description", rset->getString(1));
}

LMFUnique * LMFSeqVers::createObject() const {
  LMFSeqVers *t = new LMFSeqVers;
  t->setConnection(m_env, m_conn);
  return t;
}

std::string LMFSeqVers::fetchAllSql(Statement *stmt) const {
  std::string sql = "SELECT VERS FROM LMF_SEQ_VERS";
  stmt->setSQL(sql);
  return sql;
}

