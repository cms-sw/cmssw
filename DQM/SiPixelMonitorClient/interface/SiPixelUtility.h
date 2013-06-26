#ifndef SiPixelUtility_H
#define SiPixelUtility_H

/** \class SiPixelUtility
 * *
 *  Class that handles the SiPixel Quality Tests
 * 
 *  $Date: 2008/03/01 20:19:46 $
 *  $Revision: 1.5 $
 *  \author Petra Merkel
  */

#include <vector>
#include <fstream>
#include <string>
#include <map>
#include "TH1.h"
#include "TPaveText.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

class SiPixelUtility
{
 public:
 
 static int getMEList(std::string name, std::vector<std::string>& values);
 static bool checkME(std::string element, std::string name, std::string& full_path);
 static int getMEList(std::string name, std::string& dir_path, std::vector<std::string>& me_names);

 static void split(const std::string& str, std::vector<std::string>& tokens, 
                   const std::string& delimiters=" ");
 static void getStatusColor(int    status, int& rval, int&gval, int& bval);
 static void getStatusColor(int    status, int& icol, std::string& tag);
 static void getStatusColor(double status, int& rval, int&gval, int& bval);
 static int  getStatus(MonitorElement* me);
 
 static int computeHistoBin(std::string& module_path);
 static int computeErrorCode(DQMStore * bei, std::string& module_path);
 static int computeErrorCode(int status);
 static void fillPaveText(TPaveText* pave, std::map<std::string,std::pair<int,double> > messages); 
 static void createStatusLegendMessages(std::map<std::string,std::pair<int,double> >& messages);
 static std::map<std::string,std::string> sourceCodeMap();
 static void setDrawingOption(TH1* hist,float xlow=-1.,float xhigh=-1.);
 static std::vector<std::string> getQTestNameList(MonitorElement* me);
 
};

#endif
