
#include "RecoBTag/XMLCalibration/interface/CalibrationXML.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/dom/DOMWriter.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;


XERCES_CPP_NAMESPACE_USE

CalibrationXML::CalibrationXML() : errHandler(0), parser(0)
{

}

CalibrationXML::~CalibrationXML()
{
//TODO: delete!!!!	
if(errHandler) delete errHandler;
if(parser)  { 
              delete parser;
              XMLPlatformUtils::Terminate();
            }
}

void CalibrationXML::openFile(const std::string & xmlFileName) 
{
if(errHandler) delete errHandler;
if(parser) { delete parser; XMLPlatformUtils::Terminate(); }

 m_xmlFileName = xmlFileName;
// std::cout << "Opening.." << std::endl;
	// Initialize the XML4C2 system
	try
        {
	        XMLPlatformUtils::Initialize();
        }
	catch(const XMLException& toCatch)
	{
		std::cerr << "Error during Xerces-c Initialization.\n"
		     << "  Exception message:"
		     << XMLString::transcode(toCatch.getMessage()) << std::endl;
   abort();
//FIXME		throw GenTerminate("Error during Xerces-c Initialization.");
	}
	parser = new XercesDOMParser;
	parser->setValidationScheme(XercesDOMParser::Val_Auto);
	parser->setDoNamespaces(false);
	parser->setDoSchema(false);
	parser->setValidationSchemaFullChecking(false);
	errHandler = new HandlerBase;
	parser->setErrorHandler(errHandler);
	parser->setCreateEntityReferenceNodes(false);
	//  Parse the XML file, catching any XML exceptions that might propogate out of it.
	bool errorsOccured = false;
	try
	{
		  edm::LogInfo("XMLCalibration") << "Calibration XML: parsing " << m_xmlFileName.c_str() << std::endl;
		parser->parse(m_xmlFileName.c_str());
		int errorCount = parser->getErrorCount();
		if (errorCount > 0) errorsOccured = true;
	}
	catch (const XMLException& e)
	{
		std::cerr << "A DOM error occured during parsing\n   DOMException code: "
		     << (long unsigned int)e.getCode() << std::endl;
		errorsOccured = true;
	}
	// If the parse was successful, build the structure we want to have
	if(errorsOccured) { 
		std::cerr << "An error occured during parsing\n"
		     <<	"Please check your input with SAXCount or a similar tool.\n Exiting!\n" << std::endl; 
abort();
//FIXME		throw GenTerminate("An error occured during parsing\n Please check your input with SAXCount or a similar tool.\n Exiting!\n");
	}

	doc = parser->getDocument();
	DOMNode* n1 = doc->getFirstChild();

        while(n1)
        {
                if (n1->getNodeType() == DOMNode::ELEMENT_NODE   ) break;
                n1 = n1->getNextSibling();
        }
	
	if(strcmp("Calibration",XMLString::transcode(n1->getNodeName())))
abort();
//FIXME		throw GenTerminate("The root element in the XML Calibration file is not a Calibration element.\n This should be forbidden at the DTD level.");
	else {   edm::LogInfo("XMLCalibration")  << "Calibration found" ; }	

	m_calibrationDOM = (DOMElement *) n1;
    


}

void CalibrationXML::saveFile(const std::string & xmlFileName)
{
    DOMImplementation *	theImpl = DOMImplementationRegistry::getDOMImplementation(XMLString::transcode("Core"));
    DOMWriter         *   theSerializer = ((DOMImplementation*)theImpl)->createDOMWriter();
    theSerializer->canSetFeature(XMLUni::fgDOMWRTFormatPrettyPrint, true);
    XMLFormatTarget* myFormTarget = new LocalFileFormatTarget(XMLString::transcode(xmlFileName.c_str()));
    theSerializer->writeNode(myFormTarget, *doc);
     delete myFormTarget;
	  
}
DOMElement * CalibrationXML::addChild(DOMNode *dom,const std::string & name)
{ 
	  DOMNode *n1 = dom;
	  int level=0;
	  std::string indent="\n";
	  while(n1 && level < 100)
	  {
	   level++;
	   indent+="  ";
	   n1 = n1->getParentNode();
	  } 
	  if(dom->getFirstChild()==0)
             dom->appendChild(dom->getOwnerDocument()->createTextNode(XMLString::transcode(indent.c_str()))); 
         
	  DOMElement * child = (DOMElement *)dom->appendChild(dom->getOwnerDocument()->createElement(XMLString::transcode(name.c_str()))); 
          dom->appendChild(dom->getOwnerDocument()->createTextNode(XMLString::transcode(indent.c_str())));           
	  return child;
}  
