#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/RunCommentDat.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"

using namespace std;
using namespace oracle::occi;

RunCommentDat::RunCommentDat() {
  m_env = nullptr;
  m_conn = nullptr;
  m_writeStmt = nullptr;
  m_source = "";
  m_comment = "";
  m_time = Tm();
}

RunCommentDat::~RunCommentDat() {}

void RunCommentDat::prepareWrite() noexcept(false) {
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL(
        "INSERT INTO run_comment_dat (iov_id,  "
        "source, user_comment) "
        "VALUES (:iov_id,  "
        ":source, :user_comment)");
  } catch (SQLException& e) {
    throw(std::runtime_error("RunCommentDat::prepareWrite():  " + e.getMessage()));
  }
}

void RunCommentDat::writeDB(const EcalLogicID* ecid, const RunCommentDat* item, RunIOV* iov) noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) {
    throw(std::runtime_error("RunCommentDat::writeDB:  IOV not in DB"));
  }

  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setString(2, item->getSource());
    m_writeStmt->setString(3, item->getComment());

    m_writeStmt->executeUpdate();
  } catch (SQLException& e) {
    throw(std::runtime_error("RunCommentDat::writeDB():  " + e.getMessage()));
  }
}

void RunCommentDat::fetchData(map<EcalLogicID, RunCommentDat>* fillMap, RunIOV* iov) noexcept(false) {
  this->checkConnection();
  fillMap->clear();

  DateHandler dh(m_env, m_conn);

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) {
    //  throw(std::runtime_error("RunCommentDat::writeDB:  IOV not in DB"));
    return;
  }

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL(
        "SELECT d.comment_id, "
        "d.source, d.user_comment, d.db_timestamp "
        "FROM run_comment_dat d "
        "WHERE d.iov_id = :iov_id order by d.logic_id ");
    stmt->setInt(1, iovID);
    ResultSet* rset = stmt->executeQuery();

    std::pair<EcalLogicID, RunCommentDat> p;
    RunCommentDat dat;
    while (rset->next()) {
      p.first = EcalLogicID("Comment_order",
                            rset->getInt(1),
                            rset->getInt(1),
                            EcalLogicID::NULLID,
                            EcalLogicID::NULLID,  // comment number
                            "Comment_order");

      dat.setSource(rset->getString(2));
      dat.setComment(rset->getString(3));

      Date startDate = rset->getDate(4);
      m_time = dh.dateToTm(startDate);

      p.second = dat;
      fillMap->insert(p);
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw(std::runtime_error("RunCommentDat::fetchData():  " + e.getMessage()));
  }
}
