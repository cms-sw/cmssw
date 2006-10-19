#ifndef SiPixelUtility_H
#define SiPixelUtility_H

/** \class SiPixelUtility
 * *
 *  Class that handles the SiPixel Quality Tests
 * 
 *  $Date: 2006/10/16 18:14:27 $
 *  $Revision: 0 $
 *  \author Petra Merkel
  */

#include <vector>
#include <fstream>
#include <string>
#include <map>

class SiPixelUtility
{
 public:
 
 static int getMEList(std::string name, std::vector<std::string>& values);
 static bool checkME(std::string element, std::string name, std::string& full_path);
 static int getMEList(std::string name, std::string& dir_path, std::vector<std::string>& me_names);

 static void split(const std::string& str, std::vector<std::string>& tokens, 
             const std::string& delimiters=" ");
};

#endif
