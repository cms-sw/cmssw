#ifndef SiStripConfigParser_H
#define SiStripConfigParser_H

/** \class SiStripConfigParser
 * *
 *  Class that handles the SiStrip Quality Tests
 * 
 *  $Date: 2006/05/29 21:25:55 $
 *  $Revision: 1.1 $
 *  \author Suchandra Dutta
  */

#include "DQMServices/ClientConfig/interface/DQMParserBase.h"
#include <vector>
#include <fstream>
#include <string>
#include <map>

using namespace std;

class SiStripConfigParser : public DQMParserBase {

 public:
  

  // Constructor
  SiStripConfigParser();
  
  // Destructor
  ~SiStripConfigParser();

  // get List of MEs for TrackerMap
  bool getMENamesForTrackerMap(string& tkmap_name,vector<string>& me_names);
  bool getMENamesForSummary(string &structure_name, vector<string>& me_names);
  bool getFrequencyForTrackerMap(int& u_freq);
  bool getFrequencyForSummary(int& u_freq);
 private:
  
};

#endif
