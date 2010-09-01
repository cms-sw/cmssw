#include "FWCore/Utilities/interface/ExceptionCollector.h"

namespace edm {
  void
  ExceptionCollector::rethrow() const {
    throw exception_;
  }

  void
  ExceptionCollector::call(boost::function<void(void)> f) {
    try {
      f();
    }
    catch (cms::Exception const& e) {
      hasThrown_ = true;
      exception_ << e;
    }
    catch (std::exception const& e) {
      hasThrown_ = true;
      exception_ << e.what();
    }
    catch (...) {
      hasThrown_ = true;
      exception_ << "Unknown exception";
    }
  }
}
