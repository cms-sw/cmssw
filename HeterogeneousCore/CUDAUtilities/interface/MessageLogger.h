#ifndef HeterogeneousCore_CUDAUtilities_interface_MessageLogger_h
#define HeterogeneousCore_CUDAUtilities_interface_MessageLogger_h

#include <sstream>
#include <string>

namespace cms {
  namespace cuda {

    /**
   * This class is a temporary measure to hide C++17 constructs in
   * MessaLogger from .cu files (those are mainly files that launch
   * kernels). It will be removed once we will be able to compile .cu
   * files with C++17 capable compiler.
   */
    class MessageLogger {
    public:
      MessageLogger(std::string const& category) : category_(category) {}

      MessageLogger(std::string&& category) : category_(std::move(category)) {}

      ~MessageLogger() = default;

      MessageLogger(MessageLogger const&) = delete;
      MessageLogger(MessageLogger&&) = delete;
      MessageLogger& operator=(MessageLogger const&) = delete;
      MessageLogger& operator=(MessageLogger&&) = delete;

      template <typename T>
      MessageLogger& operator<<(T const& element) {
        message_ << element;
        return *this;
      }

    protected:
      std::string category_;
      std::stringstream message_;
    };

    class LogSystem : public MessageLogger {
    public:
      LogSystem(std::string const& category) : MessageLogger(category) {}
      LogSystem(std::string&& category) : MessageLogger(std::move(category)) {}
      ~LogSystem();
    };

    class LogAbsolute : public MessageLogger {
    public:
      LogAbsolute(std::string const& category) : MessageLogger(category) {}
      LogAbsolute(std::string&& category) : MessageLogger(std::move(category)) {}
      ~LogAbsolute();
    };

    class LogError : public MessageLogger {
    public:
      LogError(std::string const& category) : MessageLogger(category) {}
      LogError(std::string&& category) : MessageLogger(std::move(category)) {}
      ~LogError();
    };

    class LogProblem : public MessageLogger {
    public:
      LogProblem(std::string const& category) : MessageLogger(category) {}
      LogProblem(std::string&& category) : MessageLogger(std::move(category)) {}
      ~LogProblem();
    };

    class LogImportant : public MessageLogger {
    public:
      LogImportant(std::string const& category) : MessageLogger(category) {}
      LogImportant(std::string&& category) : MessageLogger(std::move(category)) {}
      ~LogImportant();
    };

    class LogWarning : public MessageLogger {
    public:
      LogWarning(std::string const& category) : MessageLogger(category) {}
      LogWarning(std::string&& category) : MessageLogger(std::move(category)) {}
      ~LogWarning();
    };

    class LogPrint : public MessageLogger {
    public:
      LogPrint(std::string const& category) : MessageLogger(category) {}
      LogPrint(std::string&& category) : MessageLogger(std::move(category)) {}
      ~LogPrint();
    };

    class LogInfo : public MessageLogger {
    public:
      LogInfo(std::string const& category) : MessageLogger(category) {}
      LogInfo(std::string&& category) : MessageLogger(std::move(category)) {}
      ~LogInfo();
    };

    class LogVerbatim : public MessageLogger {
    public:
      LogVerbatim(std::string const& category) : MessageLogger(category) {}
      LogVerbatim(std::string&& category) : MessageLogger(std::move(category)) {}
      ~LogVerbatim();
    };

  }  // namespace cuda
}  // namespace cms

#endif  // HeterogeneousCore_CUDAUtilities_interface_MessageLogger_h
