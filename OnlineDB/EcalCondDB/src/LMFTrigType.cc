#include "OnlineDB/EcalCondDB/interface/LMFTrigType.h"

using namespace std;
using namespace oracle::occi;

LMFTrigType::LMFTrigType()
{
  setClassName("LMFTrigType");
  m_stringFields["short_name"] = "";
  m_stringFields["long_name"]  = "";
}

LMFTrigType::LMFTrigType(oracle::occi::Environment* env,
			 oracle::occi::Connection* conn) : 
  LMFUnique(env, conn) {
  setClassName("LMFTrigType");
  m_stringFields["short_name"] = "";
  m_stringFields["long_name"]  = "";
}

LMFTrigType::LMFTrigType(EcalDBConnection *c) : LMFUnique(c) {
  setClassName("LMFTrigType");
  m_stringFields["short_name"] = "";
  m_stringFields["long_name"]  = "";
}

LMFTrigType::~LMFTrigType()
{
}

LMFTrigType& LMFTrigType::setName(std::string s) {
  setString("short_name", s);
  fetchID();
  return *this;
}

std::string LMFTrigType::fetchIdSql(Statement *stmt)
{
  std::string sql = "SELECT TRIG_TYPE, SNAME, LNAME FROM LMF_TRIG_TYPE_DEF "
    "WHERE "
    "SNAME   = :1";
  stmt->setSQL(sql);
  stmt->setString(1, getShortName());
  return sql;
}

std::string LMFTrigType::setByIDSql(Statement *stmt, int id) 
{
  std::string sql = "SELECT SNAME, LNAME FROM LMF_TRIG_TYPE_DEF "
    "WHERE TRIG_TYPE = :1";
  stmt->setSQL(sql);
  stmt->setInt(1, id);
  return sql;
}   

void LMFTrigType::getParameters(ResultSet *rset) {
  setString("short_name", rset->getString(1));
  setString("long_name", rset->getString(2));
}

LMFTrigType * LMFTrigType::createObject() const {
  LMFTrigType * t = new LMFTrigType();
  t->setConnection(m_env, m_conn);
  return t;
}

std::string LMFTrigType::fetchAllSql(Statement *stmt) const {
  std::string sql = "SELECT TRIG_TYPE FROM LMF_TRIG_TYPE_DEF";
  stmt->setSQL(sql);
  return sql;
}

