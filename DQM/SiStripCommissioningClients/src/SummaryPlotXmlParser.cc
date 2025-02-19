#include "DQM/SiStripCommissioningClients/interface/SummaryPlotXmlParser.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <stdexcept>

using namespace xercesc;
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
SummaryPlotXmlParser::SummaryPlotXmlParser() {
  plots_.clear();
  try { XMLPlatformUtils::Initialize(); }
  catch ( const XMLException &f ) {
    throw( std::runtime_error("Standard pool exception : Fatal Error on pool::TrivialFileCatalog") );
  }
}

// -----------------------------------------------------------------------------
//
std::vector<SummaryPlot> SummaryPlotXmlParser::summaryPlots( const sistrip::RunType& run_type ) {
  if( plots_.empty() ) {
    edm::LogWarning(mlDqmClient_)
      << "[SummaryPlotXmlParser" << __func__ << "]"
      << " You have not called the parseXML function,"
      << " or your XML file is erronious" << std::endl;
  }
  if( plots_.find( run_type ) != plots_.end() ) {
    return plots_[run_type];
  } else { return std::vector<SummaryPlot>(); }
  
}

// -----------------------------------------------------------------------------
//
void SummaryPlotXmlParser::parseXML( const std::string& f ) {
  
  plots_.clear();
  
  try {

    // Create parser and open XML document
    getDocument(f);
    
    // Retrieve root element
    DOMElement* root = this->doc()->getDocumentElement();
    if( !root ) { 
      std::stringstream ss;
      ss << "[SummaryPlotXmlParser::" << __func__ << "]"
	 << " Unable to find any elements!"
	 << " Empty xml document?...";
      throw( std::runtime_error( ss.str() ) ); 
    }

    // Check on "root" tag
    if( !XMLString::equals( root->getTagName(), XMLString::transcode(rootTag_.c_str()) ) ) {
      std::stringstream ss;
      ss << "[SummaryPlotXmlParser::" << __func__ << "]"
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
      ss << "[SummaryPlotXmlParser::" << __func__ << "]"
	 << " Unable to find any children nodes!"
	 << " Empty xml document?...";
      throw( std::runtime_error( ss.str() ) ); 
      return;
    }

//     LogTrace(mlDqmClient_) 
//       << "[SummaryPlotXmlParser::" << __func__ << "]"
//       << " Found \"" << rootTag_ << "\" tag!";
    
//     LogTrace(mlDqmClient_) 
//       << "[SummaryPlotXmlParser::" << __func__ << "]"
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
// 	  ss << "[SummaryPlotXmlParser::" << __func__ << "]"
// 	     << " Found \"" << runTypeTag_ << "\" tag!" << std::endl
// 	     << "  with tag name \"" << XMLString::transcode(element->getNodeName()) << "\"" << std::endl
// 	     << "  and attr \"" << runTypeAttr_ << "\" with value \"" << XMLString::transcode(attr) << "\"";
// 	  LogTrace(mlDqmClient_) << ss.str();
	  
	  // Retrieve nodes in xml document
	  DOMNodeList* children = node->getChildNodes();
	  if ( nodes->getLength() == 0 ) { 
	    std::stringstream ss;
	    ss << "[SummaryPlotXmlParser::" << __func__ << "]"
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
		const XMLCh* level = elem->getAttribute( XMLString::transcode(levelAttr_.c_str()) );
		const XMLCh* gran = elem->getAttribute( XMLString::transcode(granularityAttr_.c_str()) );
  
		SummaryPlot plot( XMLString::transcode(mon),
				  XMLString::transcode(pres),
				  XMLString::transcode(gran),
				  XMLString::transcode(level) );
		plots_[run_type].push_back( plot );

// 		std::stringstream ss;
// 		ss << "[SummaryPlotXmlParser::" << __func__ << "]"
// 		   << " Found \"" << summaryPlotTag_ << "\" tag!" << std::endl
// 		   << "  with tag name \"" << XMLString::transcode(elem->getNodeName()) << "\"" << std::endl
// 		   << "  and attr \"" << monitorableAttr_ << "\" with value \"" << XMLString::transcode(mon) << "\"" << std::endl
// 		   << "  and attr \"" << presentationAttr_ << "\" with value \"" << XMLString::transcode(pres) << "\"" << std::endl
// 		  //<< "  and attr \"" << viewAttr_ << "\" with value \"" << XMLString::transcode(view) << "\"" << std::endl
// 		   << "  and attr \"" << levelAttr_ << "\" with value \"" << XMLString::transcode(level) << "\"" << std::endl
// 		   << "  and attr \"" << granularityAttr_ << "\" with value \"" << XMLString::transcode(gran) << "\"";
// 		LogTrace(mlDqmClient_) << ss.str();

		// Update SummaryPlot object and push back into map
		
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
    ss << "[SummaryPlotXmlParser::" << __func__ << "]"
       << " Error parsing file: " << message << std::flush;
    XMLString::release( &message );
  }

}

// -----------------------------------------------------------------------------
//
std::ostream& operator<< ( std::ostream& os, const SummaryPlotXmlParser& parser ) {
  std::stringstream ss;
  parser.print(ss);
  os << ss.str();
  return os;
}

// -----------------------------------------------------------------------------
// 
void SummaryPlotXmlParser::print( std::stringstream& ss ) const {
  ss << "[SummaryPlotXmlParser::SummaryPlot::" << __func__ << "]" 
     << " Dumping contents of parsed XML file: " << std::endl;
  using namespace sistrip;
  typedef std::vector<SummaryPlot> Plots;
  std::map<RunType,Plots>::const_iterator irun = plots_.begin();
  for ( ; irun != plots_.end(); irun++ ) {
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
