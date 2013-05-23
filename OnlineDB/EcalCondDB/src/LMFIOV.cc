#include "OnlineDB/EcalCondDB/interface/LMFIOV.h"

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

LMFIOV& LMFIOV::setStart(const Tm &start) {
  m_iov_start = start; 
  return *this;
}

LMFIOV& LMFIOV::setStop(const Tm &stop) {
  m_iov_stop = stop;
  return *this;
}

LMFIOV& LMFIOV::setIOV(const Tm &start, const Tm &stop) {
  setStart(start);
  return setStop(stop);
}

LMFIOV& LMFIOV::setVmin(int vmin) {
  m_vmin = vmin;
  return *this;
}

LMFIOV& LMFIOV::setVmax(int vmax) {
  m_vmax = vmax;
  return *this;
}

LMFIOV& LMFIOV::setVersions(int vmin, int vmax) {
  setVmin(vmin);
  return setVmax(vmax);
}

Tm LMFIOV::getStart() const {
  return m_iov_start;
}

Tm LMFIOV::getStop() const {
  return m_iov_stop;
}

int LMFIOV::getVmin() const {
  return m_vmin;
}

int LMFIOV::getVmax() const {
  return m_vmax;
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

LMFUnique * LMFIOV::createObject() const {
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

std::string LMFIOV::writeDBSql(Statement *stmt)
{
  // check that everything has been setup
  std::string seqName = sequencePostfix(m_iov_start);
  std::string sql = "INSERT INTO LMF_IOV (IOV_ID, IOV_START, IOV_STOP, "
    "VMIN, VMAX) VALUES "
    "(lmf_iov_" + seqName + "_sq.NextVal, :1, :2, :3, :4)";
  stmt->setSQL(sql);
  DateHandler dm(m_env, m_conn);
  stmt->setDate(1, dm.tmToDate(m_iov_start));
  stmt->setDate(2, dm.tmToDate(m_iov_stop));
  stmt->setInt(3, m_vmin);
  stmt->setInt(4, m_vmax);
  if (m_debug) {
    dump();
  }
  return sql;
}
