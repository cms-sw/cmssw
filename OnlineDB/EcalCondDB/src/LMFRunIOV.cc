#include "OnlineDB/EcalCondDB/interface/LMFRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/LMFDefFabric.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"

using namespace std;
using namespace oracle::occi;

void LMFRunIOV::initialize() {
  Tm tm;
  tm.setToCurrentGMTime();

  m_intFields["lmr"] = 0;
  m_intFields["tag_id"] = 0;
  m_intFields["seq_id"] = 0;
  m_intFields["color_id"] = 0;
  m_intFields["trigType_id"] = 0;
  m_stringFields["subrun_start"] = tm.str();
  m_stringFields["subrun_end"] = tm.str();
  m_stringFields["db_timestamp"] = tm.str();
  m_stringFields["subrun_type"] = "none";
  m_className = "LMFRunIOV";
  
  _fabric = NULL;
}

LMFRunIOV::LMFRunIOV() : LMFUnique()
{
  initialize();
}

LMFRunIOV::LMFRunIOV(oracle::occi::Environment* env,
		     oracle::occi::Connection* conn) : LMFUnique(env, conn)
{
  initialize();
}

LMFRunIOV::LMFRunIOV(EcalDBConnection *c) : LMFUnique(c)
{
  initialize();
}

LMFRunIOV::LMFRunIOV(const LMFRunIOV &r) {
  initialize();
  *this = r;
}

LMFRunIOV::~LMFRunIOV()
{
  if (_fabric != NULL) {
    delete _fabric;
  }
}

LMFRunIOV& LMFRunIOV::setLMFRunTag(const LMFRunTag &tag)
{
  setInt("tag_id", tag.getID());
  return *this;
}

LMFRunIOV& LMFRunIOV::setLMFRunTag(int tag_id)
{
  setInt("tag_id", tag_id);
  return *this;
}

LMFRunTag LMFRunIOV::getLMFRunTag() const
{
  LMFRunTag rtag = LMFRunTag(m_env, m_conn);
  rtag.setByID(getInt("tag_id"));
  return rtag;
}

LMFRunIOV& LMFRunIOV::setColor(const LMFColor &color)
{
  setInt("color_id", color.getID());
  return *this;
}

LMFRunIOV& LMFRunIOV::setColor(int color_id)
{
  setInt("color_id", color_id);
  return *this;
}

void LMFRunIOV::checkFabric() {
  if (_fabric == NULL) {
    _fabric = new LMFDefFabric(m_env, m_conn);
  }
}

LMFRunIOV& LMFRunIOV::setColorIndex(int color_index)
{
  checkFabric();
  setInt("color_id", _fabric->getColorID(color_index));
  return *this;
}

LMFRunIOV& LMFRunIOV::setColor(std::string name)
{
  checkFabric();
  setInt("color_id", _fabric->getColorID(name));
  return *this;
}

LMFColor LMFRunIOV::getLMFColor() const
{
  LMFColor rcol = LMFColor(m_env, m_conn);
  rcol.setByID(getInt("color_id"));
  return rcol;
}

std::string LMFRunIOV::getColorShortName() const {
  LMFColor rcol = getLMFColor();
  return rcol.getShortName();
}

std::string LMFRunIOV::getColorLongName() const {
  LMFColor rcol = getLMFColor();
  return rcol.getLongName();
}

LMFRunIOV& LMFRunIOV::setTriggerType(LMFTrigType &trigType)
{
  setInt("trigType_id", trigType.getID());
  return *this;
}

LMFRunIOV& LMFRunIOV::setTriggerType(std::string sname)
{
  checkFabric();
  setInt("trigType_id", _fabric->getTrigTypeID(sname));
  return *this;
}

LMFRunIOV& LMFRunIOV::setTriggerType(int id) {
  setInt("trigType_id", id);
  return *this;
}

LMFTrigType LMFRunIOV::getTriggerType() const
{
  LMFTrigType rt = LMFTrigType(m_env, m_conn);
  rt.setByID(getInt("trigType_id"));
  return rt;
}

LMFRunIOV& LMFRunIOV::setLmr(int n) {
  setInt("lmr", n);
  return *this;
}

int LMFRunIOV::getLmr() const {
  return getInt("lmr");
}

LMFRunIOV& LMFRunIOV::setSubRunStart(Tm start) {
  setString("subrun_start", start.str());
  return *this;
}

Tm LMFRunIOV::getSubRunStart() const {
  Tm t;
  t.setToString(getString("subrun_start"));
  return t;
}

LMFRunIOV& LMFRunIOV::setSubRunEnd(Tm stop) {
  setString("subrun_end", stop.str());
  return *this;
}

Tm LMFRunIOV::getSubRunEnd() const {
  Tm t;
  t.setToString(getString("subrun_end"));
  return t;
}

Tm LMFRunIOV::getDBInsertionTime() const {
  Tm t;
  t.setToString(getString("db_timestamp"));
  return t;
}

LMFRunIOV& LMFRunIOV::setSubRunType(const std::string &s) {
  setString("subrun_type", s);
  return *this;
}

std::string LMFRunIOV::getSubRunType() const {
  return getString("subrun_type");
}

LMFRunIOV& LMFRunIOV::setSequence(LMFSeqDat &seq)
{
  LMFSeqDat *seqdat = new LMFSeqDat();
  *seqdat = seq;
  attach("sequence", seqdat);
  setInt("seq_id", seqdat->getID());
  return *this;
}

LMFSeqDat LMFRunIOV::getSequence() const
{
  LMFSeqDat rs = LMFSeqDat(m_env, m_conn);
  rs.setByID(getInt("seq_id"));
  return rs;
}

void LMFRunIOV::dump() const {
  LMFUnique::dump();
  std::cout << "# Fabric Address: " << _fabric << std::endl;
  if (m_debug) {
    _fabric->dump();
  }
}

std::string LMFRunIOV::fetchIdSql(Statement *stmt)
{
  std::string sql = "";
  
  sql = "SELECT LMF_IOV_ID FROM LMF_RUN_IOV WHERE SEQ_ID = :1 "
    "AND LMR = :2 ";
  if (m_intFields["tag_id"] > 0) {
    sql += "AND TAG_ID = :3";
  }
  stmt->setSQL(sql);
  stmt->setInt(1, m_intFields["seq_id"]);
  stmt->setInt(2, m_intFields["lmr"]);
  if (m_intFields["tag_id"] > 0) {
    stmt->setInt(3, m_intFields["tag_id"]);
  }
  return sql;
}

std::string LMFRunIOV::setByIDSql(Statement *stmt, int id) 
{
   DateHandler dh(m_env, m_conn);
   std::string sql = "SELECT TAG_ID, SEQ_ID, LMR, COLOR_ID, TRIG_TYPE, "
     "SUBRUN_START, SUBRUN_END, SUBRUN_TYPE, DB_TIMESTAMP FROM LMF_RUN_IOV "
     "WHERE LMF_IOV_ID = :1";
   stmt->setSQL(sql);
   stmt->setInt(1, id);
   return sql;  
}

void LMFRunIOV::getParameters(ResultSet *rset) {
  DateHandler dh(m_env, m_conn);
  setLMFRunTag(rset->getInt(1));
  LMFSeqDat *seq;
  if (m_foreignKeys.find("sequence") != m_foreignKeys.end()) {
    seq = (LMFSeqDat*)m_foreignKeys["sequence"];
    setInt("seq_id", seq->getID());
  } else {
    seq = new LMFSeqDat;
    seq->setConnection(m_env, m_conn);
    seq->setByID(rset->getInt(2));
    setInt("seq_id", seq->getID());
    delete seq;
  }
  setInt("lmr", rset->getInt(3));
  setColor(rset->getInt(4));
  setTriggerType(rset->getInt(5));
  Date start = rset->getDate(6);
  setString("subrun_start", dh.dateToTm(start).str());
  Date stop = rset->getDate(7);
  setString("subrun_end", dh.dateToTm(stop).str());
  setString("subrun_type", rset->getString(8));
  setString("db_timestamp", rset->getTimestamp(9).toText("YYYY-MM-DD HH24:MI:SS", 0));
}

bool LMFRunIOV::isValid() {
  bool ret = true;
  if (!getLMFRunTag().isValid()) {
    ret = false;
  }
  if (!getSequence().isValid()) {
    ret = false;
  }
  if (!getTriggerType().isValid()) {
    ret = false;
  }
  if ((getLmr() < 0) || (getLmr() > 92)) {
    ret = false;
  }
  if (!getLMFColor().isValid()) {
    ret = false;
  }
  // subrun start and end are by definition valid
  return ret;
}

std::string LMFRunIOV::writeDBSql(Statement *stmt) 
{
  // check that everything has been setup
  int tag_id = getInt("tag_id");
  int seq_id = getInt("seq_id");
  int color_id = getInt("color_id");
  int tt = getInt("trigType_id");
  std::string sp = sequencePostfix(getSubRunStart());
  std::string sql = "INSERT INTO LMF_RUN_IOV (LMF_IOV_ID, TAG_ID, SEQ_ID, "
    "LMR, COLOR_ID, TRIG_TYPE, SUBRUN_START, SUBRUN_END, SUBRUN_TYPE) VALUES "
    "(lmf_run_iov_" + sp + "_sq.NextVal, :1, :2, :3, :4, :5, :6, :7, :8)";
  stmt->setSQL(sql);
  DateHandler dm(m_env, m_conn);
  stmt->setInt(1, tag_id);
  stmt->setInt(2, seq_id);
  stmt->setInt(3, getInt("lmr"));
  stmt->setInt(4, color_id);
  stmt->setInt(5, tt);
  stmt->setDate(6, dm.tmToDate(getSubRunStart()));
  stmt->setDate(7, dm.tmToDate(getSubRunEnd()));
  stmt->setString(8, getSubRunType());
  return sql;
}

std::list<LMFRunIOV> LMFRunIOV::fetchBySequence(vector<int> par, 
						const std::string &sql,
						const std::string &method) 
  throw(std::runtime_error)
{
  std::list<LMFRunIOV> l;
  this->checkConnection();
  try {
    Statement *stmt = m_conn->createStatement();
    stmt->setSQL(sql);
    for (unsigned int i = 0; i < par.size(); i++) {
      stmt->setInt(i + 1, par[i]);
    }
    ResultSet *rset = stmt->executeQuery();
    while (rset->next()) {
      int lmf_iov_id = rset->getInt(1);
      LMFRunIOV iov;
      iov.setConnection(m_env, m_conn);
      iov.setByID(lmf_iov_id);
      l.push_back(iov);
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error(m_className + "::" + method + ": " +
                             e.getMessage()));
  }
  return l;
}

std::list<LMFRunIOV> LMFRunIOV::fetchBySequence(const LMFSeqDat &s) {
  int seq_id = s.getID();
  vector<int> parameters;
  parameters.push_back(seq_id);
  return fetchBySequence(parameters, "SELECT LMF_IOV_ID FROM LMF_RUN_IOV "
			 "WHERE SEQ_ID = :1", "fetchBySequence");
}

std::list<LMFRunIOV> LMFRunIOV::fetchBySequence(const LMFSeqDat &s, int lmr) {
  int seq_id = s.getID();
  vector<int> parameters;
  parameters.push_back(seq_id);
  parameters.push_back(lmr);
  return fetchBySequence(parameters, "SELECT LMF_IOV_ID FROM LMF_RUN_IOV "
			 "WHERE SEQ_ID = :1 AND LMR = :2", 
			 "fetchBySequence");
}

std::list<LMFRunIOV> LMFRunIOV::fetchBySequence(const LMFSeqDat &s, int lmr,
						int type, int color) {
  int seq_id = s.getID();
  vector<int> parameters;
  parameters.push_back(seq_id);
  parameters.push_back(lmr);
  parameters.push_back(color);
  parameters.push_back(type);
  return fetchBySequence(parameters, "SELECT LMF_IOV_ID FROM LMF_RUN_IOV "
			 "WHERE SEQ_ID = :1 AND LMR = :2 AND COLOR_ID = :3 "
			 "AND TRIG_TYPE = :4",
			 "fetchBySequence");
}

std::list<LMFRunIOV> LMFRunIOV::fetchLastBeforeSequence(const LMFSeqDat &s, 
							int lmr, int type, 
							int color) {
  int seq_id = s.getID();
  vector<int> parameters;
  parameters.push_back(seq_id);
  parameters.push_back(lmr);
  parameters.push_back(color);
  parameters.push_back(type);
  return fetchBySequence(parameters, "SELECT LMF_IOV_ID FROM (SELECT "
			 "SEQ_ID, LMF_IOV_ID FROM LMF_RUN_IOV "
			 "WHERE SEQ_ID < :1 AND LMR = :2 AND COLOR_ID = :3 "
			 "AND TRIG_TYPE = :4 ORDER BY SEQ_ID DESC) WHERE "
			 "ROWNUM <= 1",
			 "fetchBySequence");
}

LMFRunIOV& LMFRunIOV::operator=(const LMFRunIOV &r) {
  if (this != &r) {
    LMFUnique::operator=(r);
    if (r._fabric != NULL) {
      checkFabric();//      _fabric = new LMFDefFabric;
      if (m_debug) {
	_fabric->debug();
	std::cout << "COPYING INTO " << _fabric << std::endl;
      }
      *_fabric = *(r._fabric);
    }
  }
  return *this;
}
