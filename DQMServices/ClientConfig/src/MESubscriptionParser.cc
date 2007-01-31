/** \file
 *
 *  Implementation of MESubscriptionParser
 *
 *  $Date: 2006/05/09 21:28:37 $
 *  $Revision: 1.1 $
 *  \author Ilaria Segoni
 */

#include "DQMServices/ClientConfig/interface/MESubscriptionParser.h"
#include "DQMServices/ClientConfig/interface/ParserFunctions.h"

#include <stdexcept>         

using namespace xercesc;

int MESubscriptionParser::n_Instances = 0;


MESubscriptionParser::MESubscriptionParser(){
	meSubscribe.clear();
	meUnubscribe.clear();

	try { 
		if (n_Instances==0) 
		XMLPlatformUtils::Initialize();  
	}
	catch (const XMLException& e) {
		throw(std::runtime_error("Standard pool exception : Fatal Error on pool::TrivialFileCatalog"));
	}
 
	++n_Instances;
}

MESubscriptionParser::~MESubscriptionParser(){
}

bool MESubscriptionParser::parseMESubscription(){

	bool parsingErrors= this->parseFile();
	return ( parsingErrors );

}

bool MESubscriptionParser::parseFile(){
	
	DOMNodeList * meNodes = doc->getElementsByTagName(qtxml::_toDOMS("ME"));
	
  	for (unsigned int ii=0; ii< meNodes->getLength(); ++ii){
		DOMElement *argNode = dynamic_cast <DOMElement *> ( meNodes ->item(ii));
		
		std::string flag = qtxml::_toString (argNode->getAttribute (qtxml::_toDOMS ("get"))); 
		
		DOMText *meNameVal = dynamic_cast <DOMText *> (argNode->getFirstChild());
		if (!meNameVal){
			return true;
		}
	   
	if(!std::strcmp(flag.c_str(),"yes"))  meSubscribe.push_back( qtxml::_toString (meNameVal->getData()) );
	if(!std::strcmp(flag.c_str(),"no"))   meUnubscribe.push_back( qtxml::_toString (meNameVal->getData()) );

	}
     
	return false;

}

