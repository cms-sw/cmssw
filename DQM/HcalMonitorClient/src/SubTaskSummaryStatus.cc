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


void SubTaskSummaryStatus::WriteThreshold(DQMStore* dbe, std::string basedir)
{
  if (onoff==false)
    return;
  MonitorElement* me;
  const std::string prev= dbe->pwd();
  dbe->setCurrentFolder(basedir+"/"+problemDir);
  std::string name="ProblemThreshold_";
  name+=problemName;
  me = dbe->get(problemDir+"/"+name.c_str());
  if (me)
    dbe->removeElement(me->getName());
  me = dbe->bookFloat(name.c_str());
  me->Fill(thresh);
  dbe->setCurrentFolder(prev);
  return;
}
