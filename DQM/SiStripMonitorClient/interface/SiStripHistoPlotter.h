#ifndef _SiStripHistoPlotter_h_
#define _SiStripHistoPlotter_h_


#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <string>

#include "TCanvas.h"

class DQMStore;
class MonitorElement;
class TH1;

class SiStripHistoPlotter {

 public:

  SiStripHistoPlotter();
 ~SiStripHistoPlotter();

  void getNamedImageBuffer(const std::string& path, std::string& image);
  void createPlots(DQMStore* dqm_store);
  void setNewPlot(std::string& path,  std::string& option, 
                  int width, int height);
  void createStaticPlot(MonitorElement* me, const std::string& file_name);
  void createCondDBPlots(DQMStore* dqm_store);
  void setNewCondDBPlot(std::string& path,  std::string& option, 
                  int width, int height);
  bool plotsToMake() { return  ((plotList_.size() > 0) ? true : false);}
  bool condDBPlotsToMake(){ return ((condDBPlotList_.size() > 0) ? true : false);}


 private:


  struct PlotParameter { 
    std::string Path;
    std::string Option;
    int         CWidth;
    int         CHeight;
     
  };

  void fillNamedImageBuffer(TCanvas * c1, const std::string& name);
  void makePlot(DQMStore* dqm_store, const PlotParameter& par);
  void makeCondDBPlots(DQMStore* dqm_store, const PlotParameter& par);
  bool hasNamedImage(const std::string & name);
  void createDummyImage(const std::string& name);
  void getDummyImage(std::string & image);

  void setDrawingOption(TH1* hist);
  void getProjection(MonitorElement* me, TH1F* tp);


  std::map<std::string, std::string>   namedPictureBuffer_;
  std::vector<PlotParameter>           plotList_;
  std::vector<PlotParameter>           condDBPlotList_;
 
};
#endif
