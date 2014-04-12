#ifndef SiStripLayoutParser_H
#define SiStripLayoutParser_H

/** \class SiStripLayoutParser
 * *
 *  Class that handles the SiStrip Quality Tests
 * 
 *  \author Suchandra Dutta
  */

#include "DQMServices/ClientConfig/interface/DQMParserBase.h"
#include <vector>
#include <fstream>
#include <string>
#include <map>


class SiStripLayoutParser : public DQMParserBase {

 public:
  

  // Constructor
  SiStripLayoutParser();
  
  // Destructor
  ~SiStripLayoutParser();

  // Get list of Layouts for ME groups
  bool getAllLayouts(std::map< std::string, std::vector<std::string> >& me_names);

 private:
  
};

#endif
