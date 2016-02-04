#ifndef SiStripConfigParser_H
#define SiStripConfigParser_H

/** \class SiStripConfigParser
 * *
 *  Class that handles the SiStrip Quality Tests
 * 
 *  $Date: 2009/02/25 17:02:37 $
 *  $Revision: 1.5 $
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
