#include "OnlineDB/EcalCondDB/interface/LMFPrimVers.h"

using namespace std;
using namespace oracle::occi;

LMFPrimVers::LMFPrimVers()
{
  setClassName("LMFPrimVers");
  setString("description", "");
}

LMFPrimVers::LMFPrimVers(EcalDBConnection *c) : LMFUnique(c) {
  setClassName("LMFPrimVers");
  setString("description", "");
}

LMFPrimVers::LMFPrimVers(oracle::occi::Environment* env,
           oracle::occi::Connection* conn) : LMFUnique(env, conn) {
  setClassName("LMFPrimVers");
  setString("description", "");
}

LMFPrimVers::~LMFPrimVers()
{
}

std::string LMFPrimVers::fetchIdSql(Statement *stmt) {
  return "";
}

std::string LMFPrimVers::setByIDSql(Statement *stmt, int id) 
{
  std::string sql = "SELECT DESCR FROM LMF_PRIM_VERS "
    "WHERE VERS = :1";
  stmt->setSQL(sql);
  stmt->setInt(1, id);
  return sql;
}

void LMFPrimVers::getParameters(ResultSet *rset) {
  setString("description", rset->getString(1));
}

LMFUnique * LMFPrimVers::createObject() const {
  LMFPrimVers *t = new LMFPrimVers;
  t->setConnection(m_env, m_conn);
  return t;
}

std::string LMFPrimVers::fetchAllSql(Statement *stmt) const {
  std::string sql = "SELECT VERS FROM LMF_PRIM_VERS";
  stmt->setSQL(sql);
  return sql;
}

