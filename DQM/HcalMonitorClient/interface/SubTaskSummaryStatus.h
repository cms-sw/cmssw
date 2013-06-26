#ifndef GUARD_SUBTASKSUMMARYSTATUS_H
#define GUARD_SUBTASKSUMMARYSTATUS_H

#include <string>
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

class SubTaskSummaryStatus
{
 public:
  SubTaskSummaryStatus(bool onoffval=false);
  ~SubTaskSummaryStatus();
  //void SetOnOff(bool onoffval);

  void Setup(std::string problemDir, std::string problemName,
	     std::string OverName,
	     std::string id, double thresh);
  void WriteThreshold(DQMStore* dbe, std::string basedir);
  bool IsOn();

  double thresh; // value above which cell is considered bad
  // Number of bad cells
  double status[5]; // HB, HE, HO, HF, ZDC;  make private?
  double ALLstatus;

  std::string problemName; // name for the set of EtaPhi problem histograms
  std::string problemDir; // directory of problem histograms
  std::string summaryName; // name of summary Problem plot
  std::string id; // store id string ("HotCells", etc.)
  bool onoff;
};

#endif
