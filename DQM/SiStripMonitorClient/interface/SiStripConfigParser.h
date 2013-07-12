#ifndef SiStripConfigParser_H
#define SiStripConfigParser_H

/** \class SiStripConfigParser
 * *
 *  Class that handles the SiStrip Quality Tests
 * 
 *  $Date: 2007/05/02 18:02:33 $
 *  $Revision: 1.4 $
 *  \author Suchandra Dutta
  */

#include "DQMServices/ClientConfig/interface/DQMParserBase.h"
#include <vector>
#include <fstream>
#include <string>
#include <map>


class SiStripConfigParser : public DQMParserBase {

 public:
  

  // Constructor
  SiStripConfigParser();
  
  // Destructor
  ~SiStripConfigParser();

  // get List of MEs for TrackerMap
  bool getMENamesForSummary(std::map<std::string, std::string>& me_names);
  bool getFrequencyForSummary(int& u_freq);

 private:
  
};

#endif
