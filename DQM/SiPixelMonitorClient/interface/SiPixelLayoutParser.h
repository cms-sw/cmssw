#ifndef SiPixelLayoutParser_H
#define SiPixelLayoutParser_H

/** \class SiPixelLayoutParser
 * *
 *  Class that handles the SiPixel Quality Tests
 * 
 *  $Date: 2007/09/13 19:42:23 $
 *  $Revision: 1.1 $
 *  \author Suchandra Dutta
  */

#include "DQMServices/ClientConfig/interface/DQMParserBase.h"
#include <vector>
#include <fstream>
#include <string>
#include <map>


class SiPixelLayoutParser : public DQMParserBase {

 public:
  

  // Constructor
  SiPixelLayoutParser();
  
  // Destructor
  ~SiPixelLayoutParser();

  // Get list of Layouts for ME groups
  bool getAllLayouts(std::map< std::string, std::vector<std::string> >& me_names);

 private:
  
};

#endif
