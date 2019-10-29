#ifndef DQM_SiStripCommissioningClients_SummaryPlotXmlParser_H
#define DQM_SiStripCommissioningClients_SummaryPlotXmlParser_H

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryPlot.h"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>

#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>

class SummaryPlotXmlParser;

/** Debug information. */
std::ostream& operator<<(std::ostream&, const SummaryPlotXmlParser&);

/** 
    @class SummaryPlotXmlParser
    @author P.Kalavase, R.Bainbridge
    
    @brief Parses the "summary plot" xml configuration file
*/
class SummaryPlotXmlParser {
public:
  // ---------- Co(de)nstructors and consts ----------

  /** Default constructor. */
  SummaryPlotXmlParser();

  // ---------- Public interface ----------

  /** Fill the map with the required tag/names and values */
  void parseXML(const std::string& xml_file);

  /** Returns SummaryPlot objects for given commissioning task. */
  std::vector<SummaryPlot> summaryPlots(const sistrip::RunType&);

  /** Debug print method. */
  void print(std::stringstream&) const;

private:
  // ---------- Private member data ----------

  /** Container holding the SummaryPlot objects. */
  std::map<sistrip::RunType, std::vector<SummaryPlot> > plots_;

};

#endif  // DQM_SiStripCommissioningClients_SummaryPlotXmlParser_H
