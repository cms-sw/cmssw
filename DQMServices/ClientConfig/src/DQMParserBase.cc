#include "DQMServices/ClientConfig/interface/QTestNames.h"
#include "DQMServices/ClientConfig/interface/DQMParserBase.h"
#include "DQMServices/ClientConfig/interface/QTestParameterNames.h"
#include "DQMServices/ClientConfig/interface/ParserFunctions.h"

#include <stdexcept>         
/** \file
 *
 *  Implementation of DQMParserBase
 *
 *  $Date: 2006/05/04 10:27:06 $
 *  $Revision: 1.1 $
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
