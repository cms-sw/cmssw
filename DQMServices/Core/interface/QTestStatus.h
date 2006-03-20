#ifndef _QTEST_STATUS_H_
#define _QTEST_STATUS_H_

namespace dqm
{
  namespace qstatus
  {
    // the smaller the number, the less severe the message
    static const int OTHER      =  30;  // anything but 'ok','warning' or 'error'

    static const int DISABLED   =  50;   // test has been disabled
    static const int INVALID    =  60;  // problem preventing test from running
    static const int INSUF_STAT  =  70;   // insufficient statistics
    static const int DID_NOT_RUN =  90; // algorithm did not run
    
    static const int STATUS_OK  =  100;  // test was succesful
    static const int WARNING    =  200;  // test had some problems
    static const int ERROR      =  300;  // test has failed
  }


} // namespace dqm

#endif
