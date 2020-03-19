#include "FWCore/Utilities/interface/ExceptionCollector.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/ConvertException.h"

#include <exception>

namespace edm {

  class MultipleException : public cms::Exception {
  public:
    MultipleException(int iReturnValue, std::string const& iMessage)
        : cms::Exception("MultipleExceptions", iMessage), returnValue_(iReturnValue) {}

    Exception* clone() const override { return new MultipleException(*this); }

  private:
    int returnCode_() const override { return returnValue_; }

    int returnValue_;
  };

  ExceptionCollector::ExceptionCollector(std::string const& initialMessage)
      : initialMessage_(initialMessage), firstException_(), accumulatedExceptions_(), nExceptions_(0) {}

  ExceptionCollector::~ExceptionCollector() {}

  bool ExceptionCollector::hasThrown() const { return nExceptions_ > 0; }

  void ExceptionCollector::rethrow() const {
    if (nExceptions_ == 1) {
      firstException_->raise();
    } else if (nExceptions_ > 1) {
      accumulatedExceptions_->raise();
    }
  }

  void ExceptionCollector::call(std::function<void(void)> f) {
    try {
      convertException::wrap([&f]() { f(); });
    } catch (cms::Exception const& ex) {
      ++nExceptions_;
      if (nExceptions_ == 1) {
        firstException_.reset(ex.clone());
        accumulatedExceptions_.reset(new MultipleException(ex.returnCode(), initialMessage_));
      }
      *accumulatedExceptions_ << nExceptions_ << "\n" << ex.explainSelf();
    }
  }

  void ExceptionCollector::addException(cms::Exception const& exception) {
    ++nExceptions_;
    if (nExceptions_ == 1) {
      firstException_.reset(exception.clone());
      accumulatedExceptions_.reset(new MultipleException(exception.returnCode(), initialMessage_));
    }
    *accumulatedExceptions_ << "----- Exception " << nExceptions_ << " -----"
                            << "\n"
                            << exception.explainSelf();
  }
}  // namespace edm
