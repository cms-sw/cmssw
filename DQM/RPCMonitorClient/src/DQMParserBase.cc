#include "DQM/RPCMonitorClient/interface/QTestNames.h"
#include "DQM/RPCMonitorClient/interface/DQMParserBase.h"
#include "DQM/RPCMonitorClient/interface/QTestParameterNames.h"
#include "DQM/RPCMonitorClient/interface/QTestDefineDebug.h"
#include "DQM/RPCMonitorClient/interface/ParserFunctions.h"

#include <stdexcept>         
/** \file
 *
 *  Implementation of DQMParserBase
 *
 *  $Date: 2006/04/24 10:04:31 $
 *  $Revision: 1.2 $
 *  \author Ilaria Segoni
 */


DQMParserBase::DQMParserBase(){
}

DQMParserBase::~DQMParserBase(){
}


void DQMParserBase::getDocument(std::string configFile){
	
	XercesDOMParser* parser = new XercesDOMParser;     
	parser->setValidationScheme(XercesDOMParser::Val_Auto);
	parser->setDoNamespaces(false);
	parser->parse(configFile.c_str()); 
	doc = parser->getDocument();
	assert(doc);

}

int DQMParserBase::countNodes(std::string tagName){
	unsigned int tagsNum  = 
 	   doc->getElementsByTagName(qtxml::_toDOMS(tagName))->getLength();
	return tagsNum;
}
