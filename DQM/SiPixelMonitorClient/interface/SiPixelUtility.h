#ifndef SiPixelUtility_H
#define SiPixelUtility_H

/** \class SiPixelUtility
 * *
 *  Class that handles the SiPixel Quality Tests
 * 
 *  $Date: 2007/02/01 16:44:39 $
 *  $Revision: 1.2 $
 *  \author Petra Merkel
  */

#include <vector>
#include <fstream>
#include <string>
#include <map>
#include "DQMServices/Core/interface/MonitorElement.h"

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
 
};

#endif
