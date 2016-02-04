#include "OnlineDB/EcalCondDB/interface/LMFIOV.h"

/*
using namespace std;
using namespace oracle::occi;

LMFIOV::LMFIOV()
{
  //standard
  m_env = NULL;
  m_conn = NULL;
  m_className = "LMFIOV";
  m_ID = 0;
  // custom
  m_iov_start = Tm();
  m_iov_stop  = Tm();
  m_vmin      = 0;
  m_vmax      = 0;
}

LMFIOV::LMFIOV(EcalDBConnection *c)
{
  //standard
  setConnection(c->getEnv(), c->getConn());
  m_className = "LMFIOV";
  m_ID = 0;
  // custom
  m_iov_start = Tm();
  m_iov_stop  = Tm();
  m_vmin      = 0;
  m_vmax      = 0;
}

LMFIOV::~LMFIOV()
{
}

std::string LMFIOV::fetchIdSql(Statement *stmt) {
  std::string sql = "SELECT IOV_ID FROM LMF_IOV "
    "WHERE IOV_START = :1 AND IOV_STOP = :2 AND "
    "VMIN = :3 AND VMIN = :4";
  DateHandler dm(m_env, m_conn);
  stmt->setSQL(sql);
  stmt->setDate(1, dm.tmToDate(m_iov_start));
  stmt->setDate(2, dm.tmToDate(m_iov_stop));
  stmt->setInt(3, m_vmin);
  stmt->setInt(4, m_vmax);
  return sql;
}

std::string LMFIOV::setByIDSql(Statement *stmt, int id) 
{
  std::string sql = "SELECT IOV_START, IOV_STOP, VMIN, VMAX FROM LMF_IOV "
    "WHERE IOV_ID = :1";
  stmt->setSQL(sql);
  stmt->setInt(1, id);
  return sql;
}

void LMFIOV::getParameters(ResultSet *rset) {
  Date d = rset->getDate(1);
  DateHandler dh(m_env, m_conn);
  m_iov_start = dh.dateToTm(d);
  d = rset->getDate(2);
  m_iov_stop = dh.dateToTm(d);
  m_vmin = rset->getInt(3);
  m_vmax = rset->getInt(4);
}

LMFUnique * LMFIOV::createObject() {
  LMFIOV *t = new LMFIOV;
  t->setConnection(m_env, m_conn);
  return t;
}

void LMFIOV::dump() const {
  cout << "################# LMFIOV ######################" << endl;
  cout << "id : " <<  m_ID << endl;
  cout << "Start: " <<  m_iov_start.str() << endl;
  cout << "Stop : " <<  m_iov_stop.str() << endl;
  cout << "Vers.: " <<  m_vmin << " - " << m_vmax << endl;
  cout << "################# LMFIOV ######################" << endl;
}
*/
