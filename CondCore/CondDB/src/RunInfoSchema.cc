#include "CondCore/CondDB/interface/Exception.h"
#include "RunInfoSchema.h"
//
#include <openssl/sha.h>

namespace cond {

  namespace persistency {

    RUN_INFO::Table::Table(coral::ISchema& schema) : m_schema(schema) {}

    bool RUN_INFO::Table::Table::exists() { return existsTable(m_schema, tname); }

    void RUN_INFO::Table::create() {
      if (exists()) {
        throwException("RUN_INFO table already exists in this schema.", "RUN_INFO::Table::create");
      }
      TableDescription<RUN_NUMBER, START_TIME, END_TIME> descr(tname);
      descr.setPrimaryKey<RUN_NUMBER>();
      createTable(m_schema, descr.get());
    }

    bool RUN_INFO::Table::select(cond::Time_t runNumber,
                                 boost::posix_time::ptime& start,
                                 boost::posix_time::ptime& end) {
      Query<START_TIME, END_TIME> q(m_schema);
      q.addCondition<RUN_NUMBER>(runNumber);
      bool ret = false;
      for (auto r : q) {
        ret = true;
        std::tie(start, end) = r;
      }
      return ret;
    }

    cond::Time_t RUN_INFO::Table::getLastInserted(boost::posix_time::ptime& start, boost::posix_time::ptime& end) {
      cond::Time_t run = cond::time::MIN_VAL;
      Query<MAX_RUN_NUMBER> q0(m_schema);
      try {
        for (auto r : q0) {
          run = std::get<0>(r);
        }
        // cope with mis-beahviour in the sqlite plugin: no result for MAX() returns NULL
      } catch (const coral::AttributeException& e) {
        std::string message(e.what());
        if (message.find("Attempt to access data of NULL attribute") != 0)
          throw;
      }
      select(run, start, end);
      return run;
    }

    bool RUN_INFO::Table::getInclusiveRunRange(
        cond::Time_t lower,
        cond::Time_t upper,
        std::vector<std::tuple<cond::Time_t, boost::posix_time::ptime, boost::posix_time::ptime> >& runData) {
      // first find the lowest existing run >= upper
      Query<MIN_RUN_NUMBER> q0(m_schema);
      q0.addCondition<RUN_NUMBER>(upper, ">=");
      for (auto r : q0)
        upper = std::get<0>(r);
      // then find the inclusive range
      Query<RUN_NUMBER, START_TIME, END_TIME> q1(m_schema);
      q1.addCondition<RUN_NUMBER>(lower, ">=").addCondition<RUN_NUMBER>(upper, "<=");
      size_t prevSize = runData.size();
      for (auto r : q1) {
        runData.push_back(r);
      }
      return runData.size() > prevSize;
    }

    bool RUN_INFO::Table::getInclusiveTimeRange(
        const boost::posix_time::ptime& lower,
        const boost::posix_time::ptime& upper,
        std::vector<std::tuple<cond::Time_t, boost::posix_time::ptime, boost::posix_time::ptime> >& runData) {
      boost::posix_time::ptime up = upper;
      // first find the lowest existing run >= upper
      Query<START_TIME> q0(m_schema);
      q0.addCondition<START_TIME>(upper, ">=");
      bool found = q0.retrievedRows();
      if (!found)
        return false;
      Query<MIN_START_TIME> q1(m_schema);
      q1.addCondition<START_TIME>(upper, ">=");
      for (auto r : q1)
        up = std::get<0>(r);
      // then find the inclusive range
      Query<RUN_NUMBER, START_TIME, END_TIME> q2(m_schema);
      q2.addCondition<END_TIME>(lower, ">=").addCondition<START_TIME>(up, "<=");
      size_t prevSize = runData.size();
      for (auto r : q2) {
        runData.push_back(r);
      }
      return runData.size() > prevSize;
    }

    void RUN_INFO::Table::insertOne(cond::Time_t runNumber,
                                    const boost::posix_time::ptime& start,
                                    const boost::posix_time::ptime& end) {
      RowBuffer<RUN_NUMBER, START_TIME, END_TIME> dataToInsert(std::tie(runNumber, start, end));
      insertInTable(m_schema, tname, dataToInsert.get());
    }

    void RUN_INFO::Table::insert(
        const std::vector<std::tuple<cond::Time_t, boost::posix_time::ptime, boost::posix_time::ptime> >& runs) {
      BulkInserter<RUN_NUMBER, START_TIME, END_TIME> inserter(m_schema, tname);
      for (auto run : runs)
        inserter.insert(run);
      inserter.flush();
    }

    void RUN_INFO::Table::updateEnd(cond::Time_t runNumber, const boost::posix_time::ptime& end) {
      UpdateBuffer buffer;
      buffer.setColumnData<END_TIME>(std::tie(end));
      buffer.addWhereCondition<RUN_NUMBER>(runNumber);
      updateTable(m_schema, tname, buffer);
    }

    RunInfoSchema::RunInfoSchema(coral::ISchema& schema) : m_runInfoTable(schema) {}

    bool RunInfoSchema::exists() {
      if (!m_runInfoTable.exists())
        return false;
      return true;
    }

    bool RunInfoSchema::create() {
      bool created = false;
      if (!exists()) {
        m_runInfoTable.create();
        created = true;
      }
      return created;
    }

    IRunInfoTable& RunInfoSchema::runInfoTable() { return m_runInfoTable; }

  }  // namespace persistency
}  // namespace cond
