#include "DQM/SiStripCommissioningClients/interface/SummaryPlotXmlParser.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <stdexcept>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
const std::string SummaryPlotXmlParser::rootTag_ = "root";
const std::string SummaryPlotXmlParser::runTypeTag_ = "RunType";
const std::string SummaryPlotXmlParser::runTypeAttr_ = "name";
const std::string SummaryPlotXmlParser::summaryPlotTag_ = "SummaryPlot";
const std::string SummaryPlotXmlParser::monitorableAttr_ = "monitorable";
const std::string SummaryPlotXmlParser::presentationAttr_ = "presentation";
const std::string SummaryPlotXmlParser::viewAttr_ = "view";
const std::string SummaryPlotXmlParser::levelAttr_ = "level";
const std::string SummaryPlotXmlParser::granularityAttr_ = "granularity";

// -----------------------------------------------------------------------------
//
SummaryPlotXmlParser::SummaryPlotXmlParser() { plots_.clear(); }

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
void SummaryPlotXmlParser::parseXML(const std::string& filename) {
  plots_.clear();

  boost::property_tree::ptree xmltree;
  boost::property_tree::read_xml(filename, xmltree);

  auto runs = xmltree.find(rootTag_);
  if (runs == xmltree.not_found()) {
  }

  // Iterate through nodes
  for (auto& xml : xmltree) {
    if (xml.first == rootTag_) {  // find main root
      for (auto& runtype : xml.second) {
        if (runtype.first == runTypeTag_) {  // enter in the run type
          sistrip::RunType run_type =
              SiStripEnumsAndStrings::runType(runtype.second.get<std::string>("<xmlattr>." + runTypeAttr_));
          for (auto& sumplot : runtype.second) {
            if (sumplot.first == summaryPlotTag_) {
              std::string mon = sumplot.second.get<std::string>("<xmlattr>." + monitorableAttr_);
              std::string pres = sumplot.second.get<std::string>("<xmlattr>." + presentationAttr_);
              std::string level = sumplot.second.get<std::string>("<xmlattr>." + levelAttr_);
              std::string gran = sumplot.second.get<std::string>("<xmlattr>." + granularityAttr_);
              SummaryPlot plot(mon, pres, gran, level);
              plots_[run_type].push_back(plot);
            }
          }
          if (plots_[run_type].empty()) {
            std::stringstream ss;
            ss << "[SummaryPlotXmlParser::" << __func__ << "]"
               << " Unable to find any summary plot for " << runTypeTag_ << " nodes!"
               << " Empty xml summary histo block?";
            throw(std::runtime_error(ss.str()));
            return;
          }
        } else {
          std::stringstream ss;
          ss << "[SummaryPlotXmlParser::" << __func__ << "]"
             << " Unable to find any " << runTypeTag_ << " nodes!"
             << " Empty xml run-type block?";
          throw(std::runtime_error(ss.str()));
          return;
        }
      }
    } else {
      std::stringstream ss;
      ss << "[SummaryPlotXmlParser::" << __func__ << "]"
         << " Did not find \"" << rootTag_ << "\" tag! "
         << " Tag name is " << rootTag_;
      throw(std::runtime_error(ss.str()));
      return;
    }
  }
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
