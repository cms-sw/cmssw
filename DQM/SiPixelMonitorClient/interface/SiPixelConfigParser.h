#ifndef SiPixelConfigParser_H
#define SiPixelConfigParser_H

/** \class SiPixelConfigParser
 * *
 *  Class that handles the SiPixel Quality Tests
 * 
 *  $Date: 2007/02/01 16:44:06 $
 *  $Revision: 1.2 $
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
//  bool getMENamesForTrackerMap(std::string& tkmap_name,std::vector<std::string>& me_names);
//  bool getFrequencyForTrackerMap(int& u_freq);
  bool getMENamesForTree(std::string &structure_name, std::vector<std::string>& me_names);
  bool getMENamesForBarrelSummary(std::string &structure_name, std::vector<std::string>& me_names);
  bool getMENamesForEndcapSummary(std::string &structure_name, std::vector<std::string>& me_names);
  bool getFrequencyForBarrelSummary(int& u_freq);
  bool getFrequencyForEndcapSummary(int& u_freq);

 private:
  
};

#endif
