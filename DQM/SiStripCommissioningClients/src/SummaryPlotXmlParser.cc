#include "DQM/SiStripCommissioningClients/interface/SummaryPlotXmlParser.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <stdexcept>

using namespace sistrip;

SummaryPlotXmlParser::SummaryPlotXmlParser() {
}

// -----------------------------------------------------------------------------
//
std::vector<SummaryPlot> SummaryPlotXmlParser::summaryPlots(const sistrip::RunType& run_type) {
  if (plots_.empty()) {
    edm::LogWarning(mlDqmClient_) << "[SummaryPlotXmlParser" << __func__ << "]"
                                  << " You have not called the parseXML function,"
                                  << " or your XML file is erronious" << std::endl;
  }
  if (plots_.find(run_type) != plots_.end()) {
    return plots_[run_type];
  } else {
    return std::vector<SummaryPlot>();
  }
}

// -----------------------------------------------------------------------------
//
void SummaryPlotXmlParser::parseXML(const std::string& f) {
  plots_.clear();
  // TODO: implement parser based on property tree.
}

// -----------------------------------------------------------------------------
//
std::ostream& operator<<(std::ostream& os, const SummaryPlotXmlParser& parser) {
  std::stringstream ss;
  parser.print(ss);
  os << ss.str();
  return os;
}

// -----------------------------------------------------------------------------
//
void SummaryPlotXmlParser::print(std::stringstream& ss) const {
  ss << "[SummaryPlotXmlParser::SummaryPlot::" << __func__ << "]"
     << " Dumping contents of parsed XML file: " << std::endl;
  using namespace sistrip;
  typedef std::vector<SummaryPlot> Plots;
  std::map<RunType, Plots>::const_iterator irun = plots_.begin();
  for (; irun != plots_.end(); irun++) {
    ss << " RunType=\"" << SiStripEnumsAndStrings::runType(irun->first) << "\"" << std::endl;
    if (irun->second.empty()) {
      ss << " No summary plots for this RunType!";
    } else {
      Plots::const_iterator iplot = irun->second.begin();
      for (; iplot != irun->second.end(); iplot++) {
        ss << *iplot << std::endl;
      }
    }
  }
}
