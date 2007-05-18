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
void ConfigParser::SummaryPlot::checkView() {

  sistrip::View check = SiStripEnumsAndStrings::view( level_ );
  
  if ( check != view_ ) {
    std::stringstream ss;
    ss << "[ConfigParser::SummaryPlot::" << __func__ << "]"
       << " Mismatch between level_ and view_ member data!";
    if ( check != sistrip::UNKNOWN_VIEW ) {
      ss << " Changing view_ from "
	 << SiStripEnumsAndStrings::view( view_ )
	 << " to " 
	 << SiStripEnumsAndStrings::view( check ); 
      view_ = check;
    } else {
      std::string temp = SiStripEnumsAndStrings::view( view_ ) + "/" + level_;
      ss << " Changing level_ from "
	 << level_ 
	 << " to " 
	 << temp;
      level_ = temp;
    }
    //edm::LogWarning(mlDqmClient_) << ss.str();
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

    LogTrace(mlDqmClient_) 
      << "[ConfigParser::" << __func__ << "]"
      << " Found \"" << rootTag_ << "\" tag!";
    
    LogTrace(mlDqmClient_) 
      << "[ConfigParser::" << __func__ << "]"
      << " Found " << nodes->getLength()
      << " children nodes!";
    
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
	  
	  std::stringstream ss;
	  ss << "[ConfigParser::" << __func__ << "]"
	     << " Found \"" << runTypeTag_ << "\" tag!" << std::endl
	     << "  with tag name \"" << XMLString::transcode(element->getNodeName()) << "\"" << std::endl
	     << "  and attr \"" << runTypeAttr_ << "\" with value \"" << XMLString::transcode(attr) << "\"";
	  LogTrace(mlDqmClient_) << ss.str();
	  
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
		const XMLCh* view = elem->getAttribute( XMLString::transcode(viewAttr_.c_str()) );
		const XMLCh* level = elem->getAttribute( XMLString::transcode(levelAttr_.c_str()) );
		const XMLCh* gran = elem->getAttribute( XMLString::transcode(granularityAttr_.c_str()) );
  
		std::stringstream ss;
		ss << "[ConfigParser::" << __func__ << "]"
		   << " Found \"" << summaryPlotTag_ << "\" tag!" << std::endl
		   << "  with tag name \"" << XMLString::transcode(elem->getNodeName()) << "\"" << std::endl
		   << "  and attr \"" << monitorableAttr_ << "\" with value \"" << XMLString::transcode(mon) << "\"" << std::endl
		   << "  and attr \"" << presentationAttr_ << "\" with value \"" << XMLString::transcode(pres) << "\"" << std::endl
		   << "  and attr \"" << viewAttr_ << "\" with value \"" << XMLString::transcode(view) << "\"" << std::endl
		   << "  and attr \"" << levelAttr_ << "\" with value \"" << XMLString::transcode(level) << "\"" << std::endl
		   << "  and attr \"" << granularityAttr_ << "\" with value \"" << XMLString::transcode(gran) << "\"";
		LogTrace(mlDqmClient_) << ss.str();

		// Update SummaryPlot object and push back into map
		summary.reset();
		summary.mon_ = SiStripEnumsAndStrings::monitorable( XMLString::transcode(mon) );
		summary.pres_ = SiStripEnumsAndStrings::presentation( XMLString::transcode(pres) );
		summary.view_ = SiStripEnumsAndStrings::view( XMLString::transcode(view) );
		summary.gran_ = SiStripEnumsAndStrings::granularity( XMLString::transcode(gran) );
		summary.level_ = XMLString::transcode(level);
		summary.checkView();
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



