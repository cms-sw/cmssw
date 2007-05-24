#ifndef DQM_SiStripCommissioningClients_ConfigParser_H
#define DQM_SiStripCommissioningClients_ConfigParser_H

#include "DQMServices/ClientConfig/interface/DQMParserBase.h"
#include "DQMServices/ClientConfig/interface/ParserFunctions.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>

/** 
    @class ConfigParser
    @author P.Kalavase, R.Bainbridge
    
    @brief Parses the "summary plot" xml configuration file
*/
class ConfigParser : public DQMParserBase {
  
 public:

  // ---------- Constructors, destructors, structs, consts ----------

  /** Default constructor. */
  ConfigParser();

  /** Default destructor. */
  virtual ~ConfigParser() {;}

  // RunType tags and attributes
  static const std::string rootTag_;
  static const std::string runTypeTag_;
  static const std::string runTypeAttr_;

  // SummaryPlot tags and attributes
  static const std::string summaryPlotTag_;
  static const std::string monitorableAttr_;
  static const std::string presentationAttr_;
  static const std::string viewAttr_;
  static const std::string levelAttr_;
  static const std::string granularityAttr_;

  /** Class to hold SummaryPlot info. */
  class SummaryPlot {
  public:
    SummaryPlot();
    void reset();
    void print( std::stringstream& ) const;
    void checkView();
    sistrip::Monitorable mon_;
    sistrip::Presentation pres_;
    sistrip::View view_;
    sistrip::Granularity gran_;
    std::string level_;
  };

  // ---------- Public interface ----------
  
  /** Fill the map with the required tag/names and values */
  void parseXML( const std::string& xml_file );
  
  /** Returns SummaryPlot objects for given commissioning task. */
  std::vector<SummaryPlot> summaryPlots( const sistrip::RunType& );
  
  /** Debug print method. */
  void print( std::stringstream& ) const;
  
 private:

  /** Container holding the SummaryPlot objects. */
  std::map< sistrip::RunType, std::vector<SummaryPlot> > summaryPlotMap_;
  
};

// Overloaded "<<" operator for debug information 

std::ostream& operator<< ( std::ostream&, const ConfigParser& );
std::ostream& operator<< ( std::ostream&, const ConfigParser::SummaryPlot& );

#endif // DQM_SiStripCommissioningClients_ConfigParser_H


    
