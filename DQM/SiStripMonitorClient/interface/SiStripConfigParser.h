#ifndef SiStripConfigParser_H
#define SiStripConfigParser_H

/** \class SiStripConfigParser
 * *
 *  Class that handles the SiStrip Quality Tests
 * 
 *  \author Suchandra Dutta
  */

#include <vector>
#include <fstream>
#include <string>
#include <map>

#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>

class SiStripConfigParser {
public:
  // Constructor
  SiStripConfigParser();

  void getDocument(std::string filepath);

  // get List of MEs for TrackerMap
  bool getMENamesForSummary(std::map<std::string, std::string>& me_names);
  bool getFrequencyForSummary(int& u_freq);

private:
  boost::property_tree::ptree config_;
};

#endif
