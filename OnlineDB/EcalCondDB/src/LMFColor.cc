#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/LMFColor.h"

using namespace std;
using namespace oracle::occi;

LMFColor::LMFColor()
{
  m_ID = 0;
  m_className = "LMFColor";
  m_stringFields["sname"] = "none";
  m_stringFields["lname"] = "none";
  m_intFields["color"] = -1;
}

LMFColor::LMFColor(oracle::occi::Environment* env,
		   oracle::occi::Connection* conn) : LMFUnique(env, conn)
{
  m_ID = 0;
  m_className = "LMFColor";
  m_stringFields["sname"] = "none";
  m_stringFields["lname"] = "none";
  m_intFields["color"] = -1;
}

LMFColor::LMFColor(EcalDBConnection *c) : LMFUnique(c)
{
  m_ID = 0;
  m_className = "LMFColor";
  m_stringFields["sname"] = "none";
  m_stringFields["lname"] = "none";
  m_intFields["color"] = -1;
}

LMFColor::LMFColor(EcalDBConnection *c, std::string color) : LMFUnique(c)
{
  m_ID = 0;
  m_className = "LMFColor";
  m_stringFields["sname"] = "none";
  m_stringFields["lname"] = "none";
  m_intFields["color"] = -1;
  setName(color);
}

LMFColor::~LMFColor()
{
}

std::string LMFColor::fetchAllSql(Statement *stmt) const {
  std::string sql = "SELECT COLOR_ID FROM LMF_COLOR_DEF";
  stmt->setSQL(sql);
  return sql;
}

LMFUnique * LMFColor::createObject() const {
  LMFColor * n = new LMFColor;
  n->setConnection(m_env, m_conn);
  return n;
}

std::string LMFColor::fetchIdSql(Statement *stmt) {
  // the query depends on the object status
  std::string sql;
  if ((getInt("color") >= 0) && (getString("sname") != "none")) {
    sql = "SELECT COLOR_ID FROM LMF_COLOR_DEF "
          "WHERE SNAME   = :1 AND COLOR_INDEX = :2";
    stmt->setSQL(sql);
    stmt->setString(1, getShortName());
    stmt->setInt(2, getColorIndex());
  } else if (getInt("color") >= 0) {
    sql = "SELECT COLOR_ID FROM LMF_COLOR_DEF "
          "WHERE COLOR_INDEX = :1";
    stmt->setSQL(sql);
    stmt->setInt(1, getColorIndex());
  } else if (getString("sname") != "") {
    sql = "SELECT COLOR_ID FROM LMF_COLOR_DEF "
          "WHERE SNAME   = :1";
    stmt->setSQL(sql);
    stmt->setString(1, getShortName());
  } 
  return sql;
}

std::string LMFColor::setByIDSql(Statement *stmt, int id) { 
  std::string sql = "SELECT COLOR_INDEX, SNAME, LNAME "
    "FROM LMF_COLOR_DEF WHERE COLOR_ID = :1";
  stmt->setSQL(sql);
  stmt->setInt(1, id);
  return sql;
}     

void LMFColor::getParameters(ResultSet *rset) {
  setInt("color", rset->getInt(1));
  setString("sname", rset->getString(2));
  setString("lname", rset->getString(3));
}

bool LMFColor::isValid() {
  boost::ptr_list<LMFUnique> listOfValidColors = fetchAll();
  boost::ptr_list<LMFUnique>::const_iterator i = listOfValidColors.begin();
  boost::ptr_list<LMFUnique>::const_iterator e = listOfValidColors.end();
  bool ret = false;
  while (i != e) {
    LMFColor *c = (LMFColor*)&(*i);
    if (c->getShortName() == getShortName()) {
      ret = true;
      i = e;
    }
    i++;
  }
  return ret;
}
