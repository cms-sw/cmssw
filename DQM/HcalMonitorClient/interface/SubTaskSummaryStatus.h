#ifndef GUARD_SUBTASKSUMMARYSTATUS_H
#define GUARD_SUBTASKSUMMARYSTATUS_H

#include <string>

class SubTaskSummaryStatus
{
 public:
  SubTaskSummaryStatus(bool onoffval=false);
  ~SubTaskSummaryStatus();
  //void SetOnOff(bool onoffval);
  bool IsOn();

  double status[5]; // HB, HE, HO, HF, ZDC;  make private?
  double ALLstatus;
  std::string problemName; // name for the set of SJ6 problem histograms
  std::string problemDir;
  bool onoff;
 private:
  //bool onoff;
};

#endif
