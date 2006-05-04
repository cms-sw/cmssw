/** \file
 *
 *  Implementation of MESubscriptionParser
 *
 *  $Date: 2006/04/24 10:04:31 $
 *  $Revision: 1.2 $
 *  \author Ilaria Segoni
 */

#include "DQM/RPCMonitorClient/interface/MESubscriptionParser.h"
#include "DQM/RPCMonitorClient/interface/QTestDefineDebug.h"
#include "DQM/RPCMonitorClient/interface/ParserFunctions.h"

#include <stdexcept>         

int MESubscriptionParser::n_Instances = 0;


MESubscriptionParser::MESubscriptionParser(){
	meSubscribe.clear();
	meUnubscribe.clear();

	try { 
		#ifdef QT_MANAGING_DEBUG
		std::cout << "MESubscriptionParser, Xerces-c initialization Number "
		<< n_Instances<<std::endl;
		#endif
		if (n_Instances==0) 
		XMLPlatformUtils::Initialize();  
	}
	catch (const XMLException& e) {
		#ifdef QT_MANAGING_DEBUG
		std::cout << "MESubscriptionParser, Xerces-c error in initialization \n"
		<< "Exception message is:  \n"
		<< qtxml::_toString(e.getMessage()) <<std::endl;
		#endif
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
			#ifdef QT_MANAGING_DEBUG
			std::cout<<"ME List Parsing, cannot get value of "<<++ii<< "-th flag"<<std::endl;
			#endif
			return true;
		}
	   
	if(!std::strcmp(flag.c_str(),"yes"))  meSubscribe.push_back( qtxml::_toString (meNameVal->getData()) );
	if(!std::strcmp(flag.c_str(),"no"))   meUnubscribe.push_back( qtxml::_toString (meNameVal->getData()) );

	}
     
	return false;

}

