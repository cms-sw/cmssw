#include "OnlineDB/EcalCondDB/interface/LMFClsVers.h"

using namespace std;
using namespace oracle::occi;

LMFClsVers::LMFClsVers() : LMFPrimVers()
{
  setClassName("LMFClsVers");
  setString("description", "");
}

LMFClsVers::LMFClsVers(EcalDBConnection *c) : LMFPrimVers(c) {
  setClassName("LMFClsVers");
  setString("description", "");
}

LMFClsVers::LMFClsVers(oracle::occi::Environment* env,
		       oracle::occi::Connection* conn) : LMFPrimVers(env, conn) {
  setClassName("LMFClsVers");
  setString("description", "");
}

LMFClsVers::~LMFClsVers()
{
}

std::string LMFClsVers::fetchIdSql(Statement *stmt) {
  return "";
}

std::string LMFClsVers::setByIDSql(Statement *stmt, int id) 
{
  std::string sql = "SELECT DESCR FROM LMF_CLS_VERS "
    "WHERE VERS = :1";
  stmt->setSQL(sql);
  stmt->setInt(1, id);
  return sql;
}

void LMFClsVers::getParameters(ResultSet *rset) {
  setString("description", rset->getString(1));
}

LMFUnique * LMFClsVers::createObject() const {
  LMFClsVers *t = new LMFClsVers;
  t->setConnection(m_env, m_conn);
  return t;
}

std::string LMFClsVers::fetchAllSql(Statement *stmt) const {
  std::string sql = "SELECT VERS FROM LMF_CLS_VERS";
  stmt->setSQL(sql);
  return sql;
}

