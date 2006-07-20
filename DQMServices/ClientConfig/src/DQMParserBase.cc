#include "DQMServices/ClientConfig/interface/QTestNames.h"
#include "DQMServices/ClientConfig/interface/DQMParserBase.h"
#include "DQMServices/ClientConfig/interface/QTestParameterNames.h"
#include "DQMServices/ClientConfig/interface/ParserFunctions.h"

#include <stdexcept>         
/** \file
 *
 *  Implementation of DQMParserBase
 *
 *  $Date: 2006/05/09 21:28:37 $
 *  $Revision: 1.1 $
 *  \author Ilaria Segoni
 */


DQMParserBase::DQMParserBase(){
	parser=0; 
	doc=0; 
}

DQMParserBase::~DQMParserBase(){
	delete parser;
	parser=0; 
	delete doc;
	doc=0; 
}


void DQMParserBase::getDocument(std::string configFile){
	
	parser = new XercesDOMParser;     
	parser->setValidationScheme(XercesDOMParser::Val_Auto);
	parser->setDoNamespaces(false);
	parser->parse(configFile.c_str()); 
	doc = parser->getDocument();
	assert(doc);

}

void DQMParserBase::getNewDocument(std::string configFile){
	//delete doc;
	//doc =0;
	parser->resetDocumentPool();
	parser->parse(configFile.c_str()); 
	doc = parser->getDocument();
	assert(doc);

}
int DQMParserBase::countNodes(std::string tagName){
	unsigned int tagsNum  = 
 	   doc->getElementsByTagName(qtxml::_toDOMS(tagName))->getLength();
	return tagsNum;
}
