#include "DQM/HcalMonitorClient/interface/SubTaskSummaryStatus.h"

SubTaskSummaryStatus::SubTaskSummaryStatus(bool onoffval)
{
  onoff=onoffval;
  for (unsigned int i=0;i<5;++i)
    {
      status[i]=-1;  //initial status is unknown
      ALLstatus=-1;
      problemName="";
      problemDir="";
    }	   
} // constructor

SubTaskSummaryStatus::~SubTaskSummaryStatus(){}

/*
void SubTaskSummaryStatus::SetOnOff(bool onoffval)
{
  onoff=onoffval;
  return;
} // SetOnOff(bool onoffval)
*/

bool SubTaskSummaryStatus::IsOn()
{
  return onoff;
} // IsOn()

