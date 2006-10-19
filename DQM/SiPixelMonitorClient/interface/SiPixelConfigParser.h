#ifndef SiPixelConfigParser_H
#define SiPixelConfigParser_H

/** \class SiPixelConfigParser
 * *
 *  Class that handles the SiPixel Quality Tests
 * 
 *  $Date: 2006/10/16 18:14:27 $
 *  $Revision: 0.0 $
 *  \author Petra Merkel
  */

#include "DQMServices/ClientConfig/interface/DQMParserBase.h"
#include <vector>
#include <fstream>
#include <string>
#include <map>


class SiPixelConfigParser : public DQMParserBase {

 public:
  

  // Constructor
  SiPixelConfigParser();
  
  // Destructor
  ~SiPixelConfigParser();

  // get List of MEs for TrackerMap
  bool getMENamesForTrackerMap(std::string& tkmap_name,std::vector<std::string>& me_names);
  bool getMENamesForSummary(std::string &structure_name, std::vector<std::string>& me_names);
  bool getFrequencyForTrackerMap(int& u_freq);
  bool getFrequencyForSummary(int& u_freq);

 private:
  
};

#endif
