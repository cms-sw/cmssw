#include "OnlineDB/EcalCondDB/interface/LMFCorrVers.h"

using namespace std;
using namespace oracle::occi;

LMFCorrVers::LMFCorrVers() : LMFPrimVers()
{
  setClassName("LMFCorrVers");
  setString("description", "");
}

LMFCorrVers::LMFCorrVers(EcalDBConnection *c) : LMFPrimVers(c) {
  setClassName("LMFCorrVers");
  setString("description", "");
}

LMFCorrVers::LMFCorrVers(oracle::occi::Environment* env,
			 oracle::occi::Connection* conn) : 
  LMFPrimVers(env, conn)
{
  setClassName("LMFCorrVers");
  setString("description", "");
}

LMFCorrVers::~LMFCorrVers()
{
}

std::string LMFCorrVers::fetchIdSql(Statement *stmt) {
  return "";
}

std::string LMFCorrVers::setByIDSql(Statement *stmt, int id) 
{
  std::string sql = "SELECT DESCR FROM LMF_CORR_VERS "
    "WHERE VERS = :1";
  stmt->setSQL(sql);
  stmt->setInt(1, id);
  return sql;
}

void LMFCorrVers::getParameters(ResultSet *rset) {
  setString("description", rset->getString(1));
}

LMFUnique * LMFCorrVers::createObject() const {
  LMFCorrVers *t = new LMFCorrVers;
  t->setConnection(m_env, m_conn);
  return t;
}

std::string LMFCorrVers::fetchAllSql(Statement *stmt) const {
  std::string sql = "SELECT VERS FROM LMF_CORR_VERS";
  stmt->setSQL(sql);
  return sql;
}

