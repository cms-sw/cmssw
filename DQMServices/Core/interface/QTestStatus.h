#ifndef _QTEST_STATUS_H_
#define _QTEST_STATUS_H_

namespace dqm
{
  namespace qstatus
  {
    static const int STATUS_OK  =   0;  // test was succesful
    static const int WARNING    =  10;  // test had some problems
    static const int ERROR      =  20;  // test has failed

    static const int DISABLED   =  50;   // test has been disabled
    static const int INVALID    =  60;   // problem preventing test from running
    static const int INSUF_STAT =  70;   // insufficient statistics 
    static const int DID_NOT_RUN  =  90; // algorithm did not run
  }


} // namespace dqm

#endif
