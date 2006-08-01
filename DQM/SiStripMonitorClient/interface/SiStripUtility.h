#ifndef SiStripUtility_H
#define SiStripUtility_H

/** \class SiStripUtility
 * *
 *  Class that handles the SiStrip Quality Tests
 * 
 *  $Date: 2006/05/29 17:12:23 $
 *  $Revision: 1.2 $
 *  \author Suchandra Dutta
  */

#include <vector>
#include <fstream>
#include <string>
#include <map>

class SiStripUtility
{
 public:
 
 static int getMEList(std::string name, std::vector<std::string>& values);
 static bool checkME(std::string element, std::string name, std::string& full_path);
 static int getMEList(std::string name, std::string& dir_path, std::vector<std::string>& me_names);

 static void split(const std::string& str, std::vector<std::string>& tokens, 
             const std::string& delimiters=" ");
};

#endif
