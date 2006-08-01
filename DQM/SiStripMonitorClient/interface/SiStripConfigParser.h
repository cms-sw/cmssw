#ifndef SiStripConfigParser_H
#define SiStripConfigParser_H

/** \class SiStripConfigParser
 * *
 *  Class that handles the SiStrip Quality Tests
 * 
 *  $Date: 2006/06/25 13:50:52 $
 *  $Revision: 1.2 $
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
  bool getMENamesForTrackerMap(std::string& tkmap_name,std::vector<std::string>& me_names);
  bool getMENamesForSummary(std::string &structure_name, std::vector<std::string>& me_names);
  bool getFrequencyForTrackerMap(int& u_freq);
  bool getFrequencyForSummary(int& u_freq);

 private:
  
};

#endif
