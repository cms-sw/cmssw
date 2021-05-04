#ifndef SiPixelConfigParser_H
#define SiPixelConfigParser_H

/** \class SiPixelConfigParser
 * *
 *  Class that handles the SiPixel Quality Tests
 *
 *  \author Petra Merkel
 */

#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>

class SiPixelConfigParser {
public:
  // Constructor
  SiPixelConfigParser();

  void getDocument(std::string filename);

  // get List of MEs for TrackerMap
  bool getMENamesForTrackerMap(std::string &tkmap_name, std::vector<std::string> &me_names);
  bool getFrequencyForTrackerMap(int &u_freq);
  bool getMENamesForTree(std::string &structure_name, std::vector<std::string> &me_names);
  bool getMENamesForBarrelSummary(std::string &structure_name, std::vector<std::string> &me_names);
  bool getMENamesForEndcapSummary(std::string &structure_name, std::vector<std::string> &me_names);
  bool getMENamesForFEDErrorSummary(std::string &structure_name, std::vector<std::string> &me_names);
  bool getFrequencyForBarrelSummary(int &u_freq);
  bool getFrequencyForEndcapSummary(int &u_freq);
  bool getMENamesForGrandBarrelSummary(std::string &structure_name, std::vector<std::string> &me_names);
  bool getMENamesForGrandEndcapSummary(std::string &structure_name, std::vector<std::string> &me_names);
  bool getFrequencyForGrandBarrelSummary(int &u_freq);
  bool getFrequencyForGrandEndcapSummary(int &u_freq);
  bool getMessageLimitForQTests(int &u_freq);
  bool getSourceType(int &u_freq);
  bool getCalibType(int &u_freq);

private:
  boost::property_tree::ptree config_;
};

#endif
