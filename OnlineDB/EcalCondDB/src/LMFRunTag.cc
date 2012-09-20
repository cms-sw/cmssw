#include "OnlineDB/EcalCondDB/interface/LMFRunTag.h"

using namespace std;
using namespace oracle::occi;

LMFRunTag::LMFRunTag() : LMFUnique()
{
  setClassName("LMFRunTag");
  m_stringFields["gen_tag"] = "default";
  m_intFields["version"]   = 1;
}

LMFRunTag::LMFRunTag(oracle::occi::Environment* env,
		     oracle::occi::Connection* conn) : LMFUnique(env, conn)
{
  setClassName("LMFRunTag");
  m_stringFields["gen_tag"] = "default";
  m_intFields["version"]   = 1;
}

LMFRunTag::LMFRunTag(EcalDBConnection *c) : LMFUnique(c) {
  setClassName("LMFRunTag");
  m_stringFields["gen_tag"] = "default";
  m_intFields["version"]   = 1;
}

LMFRunTag::~LMFRunTag()
{
}

string LMFRunTag::getGeneralTag() const
{
  return getString("gen_tag");
}

LMFRunTag& LMFRunTag::setGeneralTag(const string &genTag)
{
  setString("gen_tag", genTag);
  return *this;
}

LMFRunTag& LMFRunTag::setVersion(int v) {
  setInt("version", v);
  return *this;
}

int LMFRunTag::getVersion() const {
  return getInt("version");
}

std::string LMFRunTag::fetchIdSql(Statement *stmt) {
  std::string sql = "SELECT tag_id FROM lmf_run_tag WHERE "
    "gen_tag    = :1 AND version = :2";
  stmt->setSQL(sql);
  stmt->setString(1, getGeneralTag());
  stmt->setInt(2, getVersion());
  return sql;
}

std::string LMFRunTag::setByIDSql(Statement *stmt, int id) 
{
  std::string sql = "SELECT gen_tag, version FROM lmf_run_tag "
    "WHERE tag_id = :1";
  stmt->setSQL(sql);
  stmt->setInt(1, id);
  return sql;
}

void LMFRunTag::getParameters(ResultSet *rset) {
  setString("gen_tag", rset->getString(1));
  setInt("version", rset->getInt(2));
}

LMFUnique * LMFRunTag::createObject() const {
  LMFRunTag *t = new LMFRunTag;
  t->setConnection(m_env, m_conn);
  return t;
}

std::string LMFRunTag::fetchAllSql(Statement *stmt) const {
  std::string sql = "SELECT TAG_ID FROM LMF_RUN_TAG";
  stmt->setSQL(sql);
  return sql;
}

bool LMFRunTag::isValid() {
  bool ret = true;
  if (getVersion() <= 0) {
    ret = false;
  }
  if (getGeneralTag().length() <= 0) {
    ret = false;
  }
  return ret;
}
