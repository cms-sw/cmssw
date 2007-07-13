#ifndef DQM_HistogramDisplayHandlers_HistogramDisplayHandler_H
#define DQM_HistogramDisplayHandlers_HistogramDisplayHandler_H 

#include "DQMServices/Components/interface/UpdateObserver.h"
#include "DQMServices/Components/interface/Updater.h"
#include "DQMServices/Components/interface/DQMBaseClient.h"
#include "DQMServices/Core/interface/MonitorUserInterface.h"

#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DQM/SiStripCommon/interface/ExtractTObject.h"

#include "TCanvas.h"
#include "toolbox/BSem.h"
#include <string>
#include <vector>

class HistogramDisplayHandler {

 public:
  
  HistogramDisplayHandler( MonitorUserInterface* mui,toolbox::BSem* b );
  ~HistogramDisplayHandler(){delete fCanvas;}
  
  void HistogramViewer(xgi::Input*, xgi::Output* ) throw ( xgi::exception::Exception );
  bool hasKey(const std::string& key);
  
  static void tokenize(const std::string& str, std::vector<std::string>& token,
		       const std::string& delimiters=" ");
  void getPair(const std::string& urlParam, const std::string& pat, std::string& key, std::string& value);
  std::string getValue(const std::string& key);
  void getHistogramList(std::vector<std::string>& hlist);
  void fillMap(const std::string& urlstr);
  void printImage(TCanvas* c1 ,xgi::Output* out);
  TCanvas* fCanvas;
  std::multimap<std::string, std::string> fReqMap;
  toolbox::BSem* fCallBack;
  MonitorUserInterface* mui_;
  
};

#endif // DQM_HistogramDisplayHandlers_HistogramDisplayHandler_H

