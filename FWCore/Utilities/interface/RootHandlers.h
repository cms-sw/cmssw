#ifndef FWCore_Utilities_RootHandlers_h
#define FWCore_Utilities_RootHandlers_h

namespace edm {
  class RootHandlers {
  public:
    RootHandlers () {}
    virtual ~RootHandlers () {}
    void disableErrorHandler() {disableErrorHandler_();}
    void enableErrorHandler() {enableErrorHandler_();}
    void enableErrorHandlerWithoutWarnings() {enableErrorHandlerWithoutWarnings_();}
  private: 
    virtual void disableErrorHandler_() = 0;
    virtual void enableErrorHandler_() = 0;
    virtual void enableErrorHandlerWithoutWarnings_() = 0;
  };
}  // end of namespace edm

#endif // InitRootHandlers_H
