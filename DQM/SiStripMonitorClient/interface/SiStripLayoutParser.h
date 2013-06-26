#ifndef SiStripLayoutParser_H
#define SiStripLayoutParser_H

/** \class SiStripLayoutParser
 * *
 *  Class that handles the SiStrip Quality Tests
 * 
 *  $Date: 2007/04/10 20:56:26 $
 *  $Revision: 1.1 $
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
