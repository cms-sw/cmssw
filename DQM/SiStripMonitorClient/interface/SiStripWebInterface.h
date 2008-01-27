#ifndef _SiStripWebInterface_h_
#define _SiStripWebInterface_h_

#include "DQMServices/WebComponents/interface/WebInterface.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
class SiStripActionExecutorQTest;
class SiStripInformationExtractor;

class SiStripWebInterface 
{
 public:
 
  enum SiStripActionType{NoAction=0, Summary=2, 
                         PlotSingleModuleHistos=5, PlotGlobalHistos=6,
                         PlotHistogramFromPath=7, PlotTkMapHistogram=8,
                         PlotHistogramFromLayout=9};

  SiStripWebInterface(DaqMonitorBEInterface* bei);
 ~SiStripWebInterface();

  
  bool readConfiguration();
 
  void handleAnalyserRequest(xgi::Input* in,xgi::Output* out, int niter);


  SiStripActionType getActionFlag() {return theActionFlag;}
  void setActionFlag(SiStripActionType flag) {theActionFlag = flag;}
  void performAction();

      
  private:

  std::string get_from_multimap(std::multimap<std::string, std::string> &mymap, std::string key);

  SiStripActionType theActionFlag;
  SiStripActionExecutorQTest* actionExecutor_;
  SiStripInformationExtractor* infoExtractor_;

  void returnReplyXml(xgi::Output * out, const std::string& name, const std::string& comment);

  std::multimap<std::string, std::string> requestMap_;

  DaqMonitorBEInterface* bei_;
 protected:


};

#endif
