#include "DQM/SiStripCommissioningClients/interface/ConfigParser.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <stdexcept>

using namespace xercesc;
using namespace sistrip;

// -----------------------------------------------------------------------------
//
const std::string ConfigParser::rootTag_ = "root";
const std::string ConfigParser::runTypeTag_ = "RunType";
const std::string ConfigParser::runTypeAttr_ = "name";
const std::string ConfigParser::summaryPlotTag_ = "SummaryPlot";
const std::string ConfigParser::monitorableAttr_ = "monitorable";
const std::string ConfigParser::presentationAttr_ = "presentation";
const std::string ConfigParser::viewAttr_ = "view";
const std::string ConfigParser::levelAttr_ = "level";
const std::string ConfigParser::granularityAttr_ = "granularity";

// -----------------------------------------------------------------------------
//
ConfigParser::ConfigParser() {
  summaryPlotMap_.clear();
  try { XMLPlatformUtils::Initialize(); }
  catch ( const XMLException &f ) {
    throw( std::runtime_error("Standard pool exception : Fatal Error on pool::TrivialFileCatalog") );
  }
}

// -----------------------------------------------------------------------------
// 
ConfigParser::SummaryPlot::SummaryPlot() :
  mon_( sistrip::UNKNOWN_MONITORABLE ),
  pres_( sistrip::UNKNOWN_PRESENTATION ),
  view_( sistrip::UNKNOWN_VIEW ),
  gran_( sistrip::UNKNOWN_GRAN ),
  level_("")
{;}

// -----------------------------------------------------------------------------
// 
void ConfigParser::SummaryPlot::reset() {
  mon_ =  sistrip::UNKNOWN_MONITORABLE;
  pres_ =  sistrip::UNKNOWN_PRESENTATION;
  view_ = sistrip::UNKNOWN_VIEW;
  gran_ =  sistrip::UNKNOWN_GRAN;
  level_ = "";
}

// -----------------------------------------------------------------------------
// 
void ConfigParser::SummaryPlot::check() {

  // Remove end "/" if it exists
  if ( !level_.empty() ) {
    std::string slash = level_.substr( level_.size()-1, 1 );
    if ( slash == sistrip::dir_ ) { level_ = level_.substr( 0, level_.size()-1 ); }
  }

  // Check view 
  sistrip::View check = SiStripEnumsAndStrings::view( level_ );
  view_ = check;
  if ( check == sistrip::UNKNOWN_VIEW || 
       check == sistrip::UNDEFINED_VIEW ) {
    edm::LogWarning(mlDqmClient_)
      << "[ConfigParser::SummaryPlot::" << __func__ << "]"
      << " Unexpected view: \"" << SiStripEnumsAndStrings::view( check );
  }

  // Add sistrip::root_ if not found
  if ( level_.find( sistrip::root_ ) == std::string::npos ) { 
    if ( check == sistrip::UNKNOWN_VIEW ) {
      level_ = 
	sistrip::root_ + sistrip::dir_ + 
	sistrip::unknownView_ + sistrip::dir_ + 
	level_; 
    } else if ( check == sistrip::UNDEFINED_VIEW ) {
      level_ = 
	sistrip::root_ + sistrip::dir_ + 
	sistrip::undefinedView_ + sistrip::dir_ + 
	level_; 
    } else { 
      level_ = 
	sistrip::root_ + sistrip::dir_ + 
	level_; 
    }
  }
  
}

// -----------------------------------------------------------------------------
//
std::vector<ConfigParser::SummaryPlot> ConfigParser::summaryPlots( const sistrip::RunType& run_type ) {
  if( summaryPlotMap_.empty() ) {
    edm::LogWarning(mlDqmClient_)
      << "[ConfigParser" << __func__ << "]"
      << " You have not called the parseXML function,"
      << " or your XML file is erronious" << std::endl;
  }
  if( summaryPlotMap_.find( run_type ) != summaryPlotMap_.end() ) {
    return summaryPlotMap_[run_type];
  } else { return std::vector<SummaryPlot>(); }
  
}

// -----------------------------------------------------------------------------
//
void ConfigParser::parseXML( const std::string& f ) {
  
  summaryPlotMap_.clear();
  ConfigParser::SummaryPlot summary;
  
  try {

    // Create parser and open XML document
    getDocument(f);
    
    // Retrieve root element
    DOMElement* root = this->doc->getDocumentElement();
    if( !root ) { 
      std::stringstream ss;
      ss << "[ConfigParser::" << __func__ << "]"
	 << " Unable to find any elements!"
	 << " Empty xml document?...";
      throw( std::runtime_error( ss.str() ) ); 
    }

    // Check on "root" tag
    if( !XMLString::equals( root->getTagName(), XMLString::transcode(rootTag_.c_str()) ) ) {
      std::stringstream ss;
      ss << "[ConfigParser::" << __func__ << "]"
	 << " Did not find \"" << rootTag_ << "\" tag! " 
	 << " Tag name is "
	 << XMLString::transcode(root->getNodeName());
      edm::LogWarning(mlDqmClient_) << ss.str();
      return;
    }
        
    // Retrieve nodes in xml document
    DOMNodeList* nodes = root->getChildNodes();
    if ( nodes->getLength() == 0 ) { 
      std::stringstream ss;
      ss << "[ConfigParser::" << __func__ << "]"
	 << " Unable to find any children nodes!"
	 << " Empty xml document?...";
      throw( std::runtime_error( ss.str() ) ); 
      return;
    }

//     LogTrace(mlDqmClient_) 
//       << "[ConfigParser::" << __func__ << "]"
//       << " Found \"" << rootTag_ << "\" tag!";
    
//     LogTrace(mlDqmClient_) 
//       << "[ConfigParser::" << __func__ << "]"
//       << " Found " << nodes->getLength()
//       << " children nodes!";
    
    // Iterate through nodes
    for( XMLSize_t inode = 0; inode < nodes->getLength(); ++inode ) {

      // Check on whether node is element
      DOMNode* node = nodes->item(inode);
      if( node->getNodeType() &&
	  node->getNodeType() == DOMNode::ELEMENT_NODE ) {
	
	DOMElement* element = dynamic_cast<DOMElement*>( node );
	if ( !element ) { continue; }

	if( XMLString::equals( element->getTagName(), 
			       XMLString::transcode(runTypeTag_.c_str()) ) ) {
	  
	  const XMLCh* attr = element->getAttribute( XMLString::transcode(runTypeAttr_.c_str()) );
	  sistrip::RunType run_type = SiStripEnumsAndStrings::runType( XMLString::transcode(attr) );
	  
// 	  std::stringstream ss;
// 	  ss << "[ConfigParser::" << __func__ << "]"
// 	     << " Found \"" << runTypeTag_ << "\" tag!" << std::endl
// 	     << "  with tag name \"" << XMLString::transcode(element->getNodeName()) << "\"" << std::endl
// 	     << "  and attr \"" << runTypeAttr_ << "\" with value \"" << XMLString::transcode(attr) << "\"";
// 	  LogTrace(mlDqmClient_) << ss.str();
	  
	  // Retrieve nodes in xml document
	  DOMNodeList* children = node->getChildNodes();
	  if ( nodes->getLength() == 0 ) { 
	    std::stringstream ss;
	    ss << "[ConfigParser::" << __func__ << "]"
	       << " Unable to find any children nodes!"
	       << " Empty xml document?...";
	    throw( std::runtime_error( ss.str() ) ); 
	    return;
	  }

	  // Iterate through nodes
	  for( XMLSize_t jnode = 0; jnode < children->getLength(); ++jnode ) {

	    // Check on whether node is element
	    DOMNode* child = children->item(jnode);
	    if( child->getNodeType() &&
		child->getNodeType() == DOMNode::ELEMENT_NODE ) {
	
	      DOMElement* elem = dynamic_cast<DOMElement*>( child );
	      if ( !elem ) { continue; }

	      if( XMLString::equals( elem->getTagName(), 
				     XMLString::transcode(summaryPlotTag_.c_str()) ) ) {
	  	
		const XMLCh* mon = elem->getAttribute( XMLString::transcode(monitorableAttr_.c_str()) );
		const XMLCh* pres = elem->getAttribute( XMLString::transcode(presentationAttr_.c_str()) );
		//const XMLCh* view = elem->getAttribute( XMLString::transcode(viewAttr_.c_str()) );
		const XMLCh* level = elem->getAttribute( XMLString::transcode(levelAttr_.c_str()) );
		const XMLCh* gran = elem->getAttribute( XMLString::transcode(granularityAttr_.c_str()) );
  
// 		std::stringstream ss;
// 		ss << "[ConfigParser::" << __func__ << "]"
// 		   << " Found \"" << summaryPlotTag_ << "\" tag!" << std::endl
// 		   << "  with tag name \"" << XMLString::transcode(elem->getNodeName()) << "\"" << std::endl
// 		   << "  and attr \"" << monitorableAttr_ << "\" with value \"" << XMLString::transcode(mon) << "\"" << std::endl
// 		   << "  and attr \"" << presentationAttr_ << "\" with value \"" << XMLString::transcode(pres) << "\"" << std::endl
// 		  //<< "  and attr \"" << viewAttr_ << "\" with value \"" << XMLString::transcode(view) << "\"" << std::endl
// 		   << "  and attr \"" << levelAttr_ << "\" with value \"" << XMLString::transcode(level) << "\"" << std::endl
// 		   << "  and attr \"" << granularityAttr_ << "\" with value \"" << XMLString::transcode(gran) << "\"";
// 		LogTrace(mlDqmClient_) << ss.str();

		// Update SummaryPlot object and push back into map
		summary.reset();
		summary.mon_ = SiStripEnumsAndStrings::monitorable( XMLString::transcode(mon) );
		summary.pres_ = SiStripEnumsAndStrings::presentation( XMLString::transcode(pres) );
		//summary.view_ = SiStripEnumsAndStrings::view( XMLString::transcode(view) );
		summary.gran_ = SiStripEnumsAndStrings::granularity( XMLString::transcode(gran) );
		summary.level_ = XMLString::transcode(level);
		summary.check();
		summaryPlotMap_[run_type].push_back(summary);
		
	      }
	    }
	  }
	  
	}
      }
    }

  }
  catch( XMLException& e ) {
    char* message = XMLString::transcode(e.getMessage());
    std::ostringstream ss;
    ss << "[ConfigParser::" << __func__ << "]"
       << " Error parsing file: " << message << std::flush;
    XMLString::release( &message );
  }

}

// -----------------------------------------------------------------------------
//
std::ostream& operator<< ( std::ostream& os, const ConfigParser& parser ) {
  std::stringstream ss;
  parser.print(ss);
  os << ss.str();
  return os;
}

// -----------------------------------------------------------------------------
// 
void ConfigParser::print( std::stringstream& ss ) const {
  ss << "[ConfigParser::SummaryPlot::" << __func__ << "]" 
     << " Dumping contents of parsed XML file: " << std::endl;
  using namespace sistrip;
  typedef std::vector<SummaryPlot> Plots;
  std::map<RunType,Plots>::const_iterator irun = summaryPlotMap_.begin();
  for ( ; irun != summaryPlotMap_.end(); irun++ ) {
    ss << " RunType=\"" 
       << SiStripEnumsAndStrings::runType( irun->first )
       << "\"" << std::endl;
    if ( irun->second.empty() ) {
      ss << " No summary plots for this RunType!";
    } else {
      Plots::const_iterator iplot = irun->second.begin();
      for ( ; iplot != irun->second.end(); iplot++ ) {
	ss << *iplot << std::endl;
      }
    }
  }
}

// -----------------------------------------------------------------------------
// 
void ConfigParser::SummaryPlot::print( std::stringstream& ss ) const {
  ss << "[ConfigParser::SummaryPlot::" << __func__ << "]" << std::endl
     << " Monitorable:  " <<  SiStripEnumsAndStrings::monitorable(mon_) << std::endl
     << " Presentation: " << SiStripEnumsAndStrings::presentation(pres_) << std::endl
     << " View:         " << SiStripEnumsAndStrings::view(view_) << std::endl
     << " TopLevelDir:  " << level_ << std::endl
     << " Granularity:  " << SiStripEnumsAndStrings::granularity(gran_) << std::endl;
}

// -----------------------------------------------------------------------------
//
std::ostream& operator<< ( std::ostream& os, const ConfigParser::SummaryPlot& summary ) {
  std::stringstream ss;
  summary.print(ss);
  os << ss.str();
  return os;
}



