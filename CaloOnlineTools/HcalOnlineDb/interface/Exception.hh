#ifndef hcal_Exception_hh_included
#define hcal_Exception_hh_included 1

#ifdef HAVE_XDAQ
#include "xcept/Exception.h"
#else
#include <exception>
#include <string>
#endif

namespace hcal {
  namespace exception {
#ifdef HAVE_XDAQ
    class Exception : public xcept::Exception {
    public:
      Exception(const std::string& name,
                const std::string& message,
                const std::string& module,
                int line,
                const std::string& function)
          : xcept::Exception(name, message, module, line, function) {}

      Exception(const std::string& name,
                const std::string& message,
                const std::string& module,
                int line,
                const std::string& function,
                xcept::Exception& e)
          : xcept::Exception(name, message, module, line, function, e) {}
    };
#else
    class Exception : public std::exception {
    public:
      Exception(const std::string& name,
                const std::string& message,
                const std::string& module,
                int line,
                const std::string& function)
          : Exception(message) {}
      Exception(const std::string& message) : message_(message) {}

      const char* what() const throw() override { return message_.c_str(); }

    private:
      std::string message_;
    };

#define XCEPT_RAISE(EXCEPTION, MSG) throw EXCEPTION(#EXCEPTION, MSG, __FILE__, __LINE__, __FUNCTION__)

#endif
  }  // namespace exception
}  // namespace hcal

#endif  // hcal_Exception_hh_included
