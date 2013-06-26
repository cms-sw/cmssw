#include "FWCore/Utilities/interface/ExceptionCollector.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/ConvertException.h"

#include <exception>

namespace edm {

  ExceptionCollector::ExceptionCollector(std::string const& initialMessage) :
    initialMessage_(initialMessage),
    firstException_(),
    accumulatedExceptions_(),
    nExceptions_(0) {
  }

  ExceptionCollector::~ExceptionCollector() {
  }

  bool
  ExceptionCollector::hasThrown() const {
    return nExceptions_ > 0;
  }

  void
  ExceptionCollector::rethrow() const {
    if (nExceptions_ == 1) {
      firstException_->raise();
    }
    else if (nExceptions_ > 1) {
      accumulatedExceptions_->raise();
    }
  }

  void
  ExceptionCollector::call(boost::function<void(void)> f) {
    try {
      try {
        f();
      }
      catch (cms::Exception& e) { throw; }
      catch (std::bad_alloc& bda) { convertException::badAllocToEDM(); }
      catch (std::exception& e) { convertException::stdToEDM(e); }
      catch (std::string& s) { convertException::stringToEDM(s); }
      catch (char const* c) { convertException::charPtrToEDM(c); }
      catch (...) { convertException::unknownToEDM(); }
    }      
    catch (cms::Exception const& ex) {
      ++nExceptions_;
      if (nExceptions_ == 1) {
        firstException_.reset(ex.clone());
        accumulatedExceptions_.reset(new cms::Exception("MultipleExceptions", initialMessage_));
      }
      *accumulatedExceptions_ << nExceptions_ << "\n"
                              << ex.explainSelf();
    }
  }

  void
  ExceptionCollector::addException(cms::Exception const& exception) {
    ++nExceptions_;
    if (nExceptions_ == 1) {
      firstException_.reset(exception.clone());
      accumulatedExceptions_.reset(new cms::Exception("MultipleExceptions", initialMessage_));
    }
    *accumulatedExceptions_ << "----- Exception " << nExceptions_ << " -----"
                            << "\n"
                            << exception.explainSelf();
  }
}
