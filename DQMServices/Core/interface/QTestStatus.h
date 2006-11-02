#ifndef _QTEST_STATUS_H_
#define _QTEST_STATUS_H_

/** Numeric constants for quality test results
    the smaller the number, the less severe the message */
namespace dqm
{
  namespace qstatus
  {
    /// anything but 'ok','warning' or 'error'
    static const int OTHER      =  30;  
    
    /// test has been disabled
    static const int DISABLED   =  50;   
    /// problem preventing test from running
    static const int INVALID    =  60;  
    /// insufficient statistics
    static const int INSUF_STAT  =  70;   
    /// algorithm did not run
    static const int DID_NOT_RUN =  90; 
    
    /// test was succesful
    static const int STATUS_OK  =  100;  
    /// test had some problems
    static const int WARNING    =  200;  
    /// test has failed
    static const int ERROR      =  300;  
  }


} // namespace dqm

#endif
