#ifndef FWCore_MessageLogger_MessageSender_h
#define FWCore_MessageLogger_MessageSender_h

#include "FWCore/MessageLogger/interface/ELseverityLevel.h"
#include "FWCore/MessageLogger/interface/ErrorObj.h"

#include <memory>

#include <map>

namespace edm {

  class MessageSender {
    struct ErrorObjDeleter {
      ErrorObjDeleter() {}
      void operator()(ErrorObj* errorObjPtr);
    };

  public:
    // ---  birth/death:
    MessageSender() = default;
    MessageSender(messagelogger::ELseverityLevel const& sev,
                  std::string_view id,
                  bool verbatim = false,
                  bool suppressed = false);
    MessageSender(MessageSender&&) = default;
    MessageSender(MessageSender const&) = default;
    MessageSender& operator=(MessageSender&&) = default;
    MessageSender& operator=(MessageSender const&) = default;
    ~MessageSender();

    // ---  stream out the next part of a message:
    template <class T>
    MessageSender& operator<<(T const& t) {
      if (valid())
        (*errorobj_p) << t;
      return *this;
    }

    template <typename... Args>
    MessageSender& format(std::string_view fmt, Args const&... args) {
      if (valid())
        errorobj_p->format(fmt, args...);
      return *this;
    }

    template <typename... Args>
    MessageSender& printf(std::string_view fmt, Args const&... args) {
      if (valid())
        errorobj_p->printf(fmt, args...);
      return *this;
    }

    bool valid() const noexcept { return errorobj_p != nullptr; }

  private:
    // data:
    std::shared_ptr<ErrorObj> errorobj_p;

  };  // MessageSender

}  // namespace edm

#endif  // FWCore_MessageLogger_MessageSender_h
