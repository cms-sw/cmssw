#ifndef _SiStripWebInterface_h_
#define _SiStripWebInterface_h_

#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "xgi/Method.h"
#include "xdata/UnsignedLong.h"
#include "cgicc/HTMLClasses.h"

#include "xdaq/Application.h"
#include "xgi/Utils.h"
#include "xgi/Method.h"

class DQMStore;
class SiStripActionExecutorQTest;
class SiStripInformationExtractor;
class SiStripDetCabling;

class SiStripWebInterface 
{
 public:

 
  enum SiStripActionType{NoAction=0, Summary=2, 
                         PlotSingleModuleHistos=5, PlotGlobalHistos=6,
                         PlotHistogramFromPath=7, PlotTkMapHistogram=8,
                         PlotHistogramFromLayout=9,
                         CreatePlots=10};

  SiStripWebInterface(DQMStore* dqm_store);
 ~SiStripWebInterface();

  
 
  //  void setCabling(const edm::ESHandle<SiStripDetCabling>& detcabling) { detCabling_ = detcabling;}

  void handleAnalyserRequest(xgi::Input* in,xgi::Output* out, const edm::ESHandle<SiStripDetCabling>& detcabling, int niter);


  SiStripActionType getActionFlag() {return theActionFlag;}
  void setActionFlag(SiStripActionType flag) {theActionFlag = flag;}
  void performAction();
  unsigned int getNumberOfConDBPlotRequest(){ return condDBRequestList_.size();}
  void getConDBPlotParameters(unsigned int ival, uint32_t &det_id, std::string& subdet_type, 
                              uint32_t&   subdet_side, uint32_t& layer_number);    
  void clearConDBPlotRequests() { condDBRequestList_.clear(); }

  std::string getTkMapType() { return TkMapType;}
      
  private:

  std::string get_from_multimap(std::multimap<std::string, std::string> &mymap, std::string key);

  SiStripActionType theActionFlag;
  SiStripActionExecutorQTest* actionExecutor_;
  SiStripInformationExtractor* infoExtractor_;

  void returnReplyXml(xgi::Output * out, const std::string& name, const std::string& comment);

  std::multimap<std::string, std::string> requestMap_;

  DQMStore* dqmStore_;

  bool condDBFlag_;


  struct CondDBPlotParameter { 
    uint32_t    detId;
    std::string type;
    uint32_t    side;
    uint32_t    layer;    
     
  };

  std::vector<CondDBPlotParameter> condDBRequestList_;
  std::string TkMapType;
 
  protected:


};

#endif
