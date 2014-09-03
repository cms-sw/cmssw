#ifndef FWCore_Utilities_ExceptionCollector_h
#define FWCore_Utilities_ExceptionCollector_h

/**
ExceptionCollector is a utility class that can be used to make sure that
each function or functor in a sequence of calls is invoked even if 
a previous function throws.  Each function/functor must take no arguments
and return a void.  std::bind can be used to convert a function
taking arguments into a function taking no arguments.
The exception strings are saved in a cms::Exception for optional rethrow.

Here is an example:

ExceptionCollector c("initialMessage");

c.call(std::bind(&MyClass::myFunction, myClassPtr));
c.call(std::bind(&MyClass::myOtherFunction, myClassPtr, myArgPtr));
c.call(std::bind(&myFreeFunction, myArgPtr));
if (c.hasThrown()) c.rethrow();

This insures that all three functions will be called before any exception is thrown.
**/

#include <functional>
#include <memory>
#include <string>

namespace cms {
  class Exception;
}

namespace edm {
  class ExceptionCollector {
  public:
    ExceptionCollector(std::string const& initialMessage);
    ~ExceptionCollector();
    bool hasThrown() const;
    void rethrow() const;
    void call(std::function<void(void)>);
    void addException(cms::Exception const& exception);

  private:
    std::string initialMessage_;
    std::auto_ptr<cms::Exception> firstException_;
    std::auto_ptr<cms::Exception> accumulatedExceptions_;
    int nExceptions_;
  };
}

#endif
