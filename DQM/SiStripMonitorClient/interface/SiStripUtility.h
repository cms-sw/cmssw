#ifndef SiStripUtility_H
#define SiStripUtility_H

/** \class SiStripUtility
 * *
 *  Class that handles the SiStrip Quality Tests
 * 
 *  $Date: 2006/08/01 18:14:27 $
 *  $Revision: 1.3 $
 *  \author Suchandra Dutta
  */

#include <vector>
#include <fstream>
#include <string>
#include <map>
#include "DQMServices/Core/interface/MonitorElement.h"


class SiStripUtility
{
 public:
 
 static int getMEList(std::string name, std::vector<std::string>& values);
 static bool checkME(std::string element, std::string name, std::string& full_path);
 static int getMEList(std::string name, std::string& dir_path, std::vector<std::string>& me_names);

 static void split(const std::string& str, std::vector<std::string>& tokens, 
             const std::string& delimiters=" ");
 static void getStatusColor(int status, int& rval, int&gval, int& bval);
 static void getStatusColor(int status, int& icol, std::string& tag);
 static int getStatus(MonitorElement* me);
};

#endif
