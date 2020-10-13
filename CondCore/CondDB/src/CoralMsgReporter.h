#ifndef CondCore_CondDB_CoralMsgReporter_h
#define CondCore_CondDB_CoralMsgReporter_h

//#include "CondCore/CondDB/interface/Logger.h"

#include <mutex>
#include "CoralBase/MessageStream.h"

namespace cond {

  namespace persistency {

    class Logger;

    class CoralMsgReporter : public coral::IMsgReporter {
    public:
      // Empty ctr is suppressed
      CoralMsgReporter() = delete;

      /// Default constructor
      explicit CoralMsgReporter(Logger& logger);

      /// Destructor
      ~CoralMsgReporter() override {}

      /// Release reference to reporter
      void release() override { delete this; }

      /// Access output level
      coral::MsgLevel outputLevel() const override;

      /// Modify output level
      void setOutputLevel(coral::MsgLevel lvl) override;

      /// Report a message
      void report(int lvl, const std::string& src, const std::string& msg) override;

    private:
      // the destination of the streams...
      Logger& m_logger;

      /// The current message level threshold
      coral::MsgLevel m_level;

      /// Use a different format output
      size_t m_format;

      /// The mutex lock
      std::recursive_mutex m_mutex;
    };

  }  // namespace persistency

}  // namespace cond
#endif
