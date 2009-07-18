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

void SubTaskSummaryStatus::Setup(std::string Dir, 
				 std::string Name,
				 std::string OverName,
				 std::string ID, 
				 double t=0.)
{
  problemDir=Dir; // directory where depth histos are stored
  problemName=Name; // base name of depth histos
  summaryName=OverName; // name of summary Problem plot (including directory)
  id=ID;
  thresh=t;
}
