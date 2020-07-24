#include "CondCore/CondDB/interface/RunInfoEditor.h"
#include "CondCore/CondDB/interface/Utils.h"
#include "SessionImpl.h"
//

namespace cond {

  namespace persistency {

    // implementation details. holds only data.
    class RunInfoEditorData {
    public:
      explicit RunInfoEditorData() : runBuffer(), updateBuffer() {}
      // update buffers
      std::vector<std::tuple<cond::Time_t, boost::posix_time::ptime, boost::posix_time::ptime> > runBuffer;
      std::vector<std::tuple<cond::Time_t, boost::posix_time::ptime, boost::posix_time::ptime> > updateBuffer;
    };

    RunInfoEditor::RunInfoEditor() : m_data(), m_session() {}

    RunInfoEditor::RunInfoEditor(const std::shared_ptr<SessionImpl>& session)
        : m_data(new RunInfoEditorData), m_session(session) {}

    RunInfoEditor::RunInfoEditor(const RunInfoEditor& rhs) : m_data(rhs.m_data), m_session(rhs.m_session) {}

    RunInfoEditor& RunInfoEditor::operator=(const RunInfoEditor& rhs) {
      m_data = rhs.m_data;
      m_session = rhs.m_session;
      return *this;
    }

    void RunInfoEditor::init() {
      if (m_data.get()) {
        checkTransaction("RunInfoEditor::init");
        if (!m_session->runInfoSchema().exists())
          m_session->runInfoSchema().create();
      }
    }

    cond::Time_t RunInfoEditor::getLastInserted() {
      if (m_data.get()) {
        checkTransaction("RunInfoEditor::getLastInserted");
        boost::posix_time::ptime start, end;
        return m_session->runInfoSchema().runInfoTable().getLastInserted(start, end);
      }
      return cond::time::MIN_VAL;
    }

    void RunInfoEditor::insert(cond::Time_t runNumber,
                               const boost::posix_time::ptime& start,
                               const boost::posix_time::ptime& end) {
      if (m_data.get())
        m_data->runBuffer.push_back(std::tie(runNumber, start, end));
    }

    void RunInfoEditor::insertNew(cond::Time_t runNumber,
                                  const boost::posix_time::ptime& start,
                                  const boost::posix_time::ptime& end) {
      if (m_data.get())
        m_data->updateBuffer.push_back(std::tie(runNumber, start, end));
    }

    size_t RunInfoEditor::flush() {
      size_t ret = 0;
      if (m_data.get()) {
        checkTransaction("RunInfoEditor::flush");
        m_session->runInfoSchema().runInfoTable().insert(m_data->runBuffer);
        ret += m_data->runBuffer.size();
        for (auto update : m_data->updateBuffer) {
          cond::Time_t newRun = std::get<0>(update);
          boost::posix_time::ptime& newRunStart = std::get<1>(update);
          boost::posix_time::ptime& newRunEnd = std::get<2>(update);
          boost::posix_time::ptime existingRunStart;
          boost::posix_time::ptime existingRunEnd;
          if (m_session->runInfoSchema().runInfoTable().select(newRun, existingRunStart, existingRunEnd)) {
            if (newRunStart != existingRunStart) {
              std::stringstream msg;
              msg << "Attempt to update start time of existing run " << newRun;
              throwException(msg.str(), "RunInfoEditor::flush");
            }
            if (existingRunEnd == newRunEnd) {
              // nothing to do
              continue;
            } else {
              if (existingRunEnd != existingRunStart) {
                std::stringstream msg;
                msg << "Attempt to update end time of existing run " << newRun;
                throwException(msg.str(), "RunInfoEditor::flush");
              }
            }
            m_session->runInfoSchema().runInfoTable().updateEnd(newRun, newRunEnd);
          } else {
            m_session->runInfoSchema().runInfoTable().insertOne(newRun, newRunStart, newRunEnd);
          }
          ret++;
        }
        m_data->runBuffer.clear();
        m_data->updateBuffer.clear();
      }
      return ret;
    }

    void RunInfoEditor::checkTransaction(const std::string& ctx) {
      if (!m_session.get())
        throwException("The session is not active.", ctx);
      if (!m_session->isTransactionActive(false))
        throwException("The transaction is not active.", ctx);
    }

  }  // namespace persistency
}  // namespace cond
