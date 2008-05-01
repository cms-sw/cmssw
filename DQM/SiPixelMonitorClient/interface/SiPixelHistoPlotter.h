#ifndef _SiPixelHistoPlotter_h_
#define _SiPixelHistoPlotter_h_


#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <string>

#include "TCanvas.h"

class DQMStore;
class MonitorElement;
class TH1;

class SiPixelHistoPlotter {

 public:

  SiPixelHistoPlotter();
 ~SiPixelHistoPlotter();

  void getNamedImageBuffer(const std::string& path, std::string& image);
  void createPlots(DQMStore* bei);
  void setNewPlot(std::string& path,  std::string& option, 
                  int width, int height);
  void createStaticPlot(MonitorElement* me, const std::string& file_name);

 private:


  struct PlotParameter { 
    std::string Path;
    std::string Option;
    int         CWidth;
    int         CHeight;
     
  };

  void fillNamedImageBuffer(TCanvas * c1, const std::string& name);
  void makePlot(DQMStore* bei, const PlotParameter& par);
  bool hasNamedImage(const std::string & name);
  void createDummyImage(const std::string& name);
  void setDrawingOption(TH1* hist);

  void setSubDetAxisDrawing(   std::string                                detector, 
                               TH1F                                     * histo);
  void setLines(               MonitorElement                           * me, 
                               std::string                              & meName, 
			       double                                   & ymin, 
			       double                                   & ymax, 
			       double                                   & warning, 
			       double                                   & error, 
			       double                                   & channelFraction);

  std::map<std::string, std::string>   namedPictureBuffer_;
  std::vector<PlotParameter>           plotList_;
 
};
#endif
