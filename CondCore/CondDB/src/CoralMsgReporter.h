#ifndef CondCore_CondDB_CoralMsgReporter_h
#define CondCore_CondDB_CoralMsgReporter_h

//#include "CondCore/CondDB/interface/Logger.h"

#include <mutex>
#include <set>
#include "CoralBase/MessageStream.h"

namespace cond {

  namespace persistency {

    class Logger;

    class MsgDispatcher {
    public:
      MsgDispatcher() = delete;
      explicit MsgDispatcher(Logger& logger);
      virtual ~MsgDispatcher() {}

      void unsubscribe();

      bool hasRecipient();
      Logger& recipient();

    private:
      Logger* m_recipient = nullptr;
    };

    class CoralMsgReporter : public coral::IMsgReporter {
    public:
      /// Default constructor
      CoralMsgReporter();

      /// Destructor
      ~CoralMsgReporter() override{};

      /// Release reference to reporter
      void release() override { delete this; }

      /// Access output level
      coral::MsgLevel outputLevel() const override;

      /// Modify output level
      void setOutputLevel(coral::MsgLevel lvl) override;

      /// Report a message
      void report(int lvl, const std::string& src, const std::string& msg) override;

      void subscribe(Logger& logger);

    private:
      // the destination of the streams...
      std::shared_ptr<MsgDispatcher> m_dispatcher;

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
