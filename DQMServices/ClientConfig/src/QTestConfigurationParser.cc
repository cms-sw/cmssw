#include "DQMServices/ClientConfig/interface/QTestConfigurationParser.h"
#include "DQMServices/ClientConfig/interface/QTestParameterNames.h"
#include "DQMServices/ClientConfig/interface/ParserFunctions.h"
#include <cstring>
#include <stdexcept>         
/** \file
 *
 *  Implementation of QTestConfigurationParser
 *
 *  $Date: 2013/04/01 09:47:14 $
 *  $Revision: 1.7 $
 *  \author Ilaria Segoni
 */
using namespace xercesc;

int QTestConfigurationParser::s_numberOfInstances = 0;


QTestConfigurationParser::QTestConfigurationParser(){
        
	qtestParamNames=new QTestParameterNames();

	try { 
		if (s_numberOfInstances==0) 
		XMLPlatformUtils::Initialize();  
	}
	catch (const XMLException& e) {
		throw(std::runtime_error("Standard pool exception : Fatal Error on pool::TrivialFileCatalog"));
	}
 
	++s_numberOfInstances;
}

QTestConfigurationParser::~QTestConfigurationParser(){
	delete qtestParamNames;
	qtestParamNames = 0;
}

bool QTestConfigurationParser::parseQTestsConfiguration(){
	testsToDisable.clear();
	testsRequested.clear();
	mapMonitorElementTests.clear();
	bool qtErrors= this->qtestsConfig();
	bool meErrors= this->monitorElementTestsMap();
	return (qtErrors||meErrors);

}

bool QTestConfigurationParser::qtestsConfig(){
	
	std::string testActivationOFF="false";

	unsigned int qtestTagsNum  = 
	  doc()->getElementsByTagName(qtxml::_toDOMS("QTEST"))->getLength();

	for (unsigned int i=0; i<qtestTagsNum; ++i){
		/// Get Node
		DOMNode* qtestNode = 
		  doc()->getElementsByTagName(qtxml::_toDOMS("QTEST"))->item(i);
	
	
		///Get QTEST name
		if (! qtestNode){
			return true;
		}
		DOMElement* qtestElement = static_cast<DOMElement *>(qtestNode);          
		if (! qtestElement){
			return true;		 
		}
		std::string qtestName = qtxml::_toString (qtestElement->getAttribute (qtxml::_toDOMS ("name"))); 
		std::string activate = qtxml::_toString (qtestElement->getAttribute (qtxml::_toDOMS ("activate"))); 
		if(!std::strcmp(activate.c_str(),testActivationOFF.c_str())) {
			testsToDisable.push_back(qtestName);
		}else{	
							
			///Get Qtest TYPE
			DOMNodeList *typeNodePrefix 
		  	= qtestElement->getElementsByTagName (qtxml::_toDOMS ("TYPE"));
	     	     
			if (typeNodePrefix->getLength () != 1)return true;
			       
	     
			DOMElement *prefixNode = dynamic_cast <DOMElement *> (typeNodePrefix->item (0));
			if (!prefixNode)return true;

			DOMText *prefixText = dynamic_cast <DOMText *> (prefixNode->getFirstChild());
			if (!prefixText)return true;			
	
			std::string qtestType = qtxml::_toString (prefixText->getData ());

			testsRequested[qtestName]=  this->getParams(qtestElement, qtestType);
		
			if( this->checkParameters(qtestName, qtestType)) return true;
		}
	
 	} //loop on qtestTagsNum
 
	return false;
 
}

std::map<std::string, std::string> QTestConfigurationParser::getParams(DOMElement* qtestElement, std::string qtestType){
	
	std::map<std::string, std::string> paramNamesValues;
	paramNamesValues["type"]=qtestType;
	
	DOMNodeList *arguments = qtestElement->getElementsByTagName (qtxml::_toDOMS ("PARAM"));
	
	for (unsigned int i=0; i<arguments->getLength(); ++i){
		DOMElement *argNode = dynamic_cast <DOMElement *> ( arguments ->item(i));
		std::string regExp = qtxml::_toString (argNode->getAttribute (qtxml::_toDOMS ("name"))); 
		DOMText *argText = dynamic_cast <DOMText *> (argNode->getFirstChild());
		if (!argText){
			break;
		}
	   
		std::string regExpValue = qtxml::_toString (argText->getData());
		paramNamesValues[regExp]=regExpValue;
	}
        
	return paramNamesValues;

}

bool QTestConfigurationParser::checkParameters(std::string qtestName, std::string qtestType){
	
	std::vector<std::string> paramNames=qtestParamNames->getTestParamNames(qtestType);
        // commenting out as does not seem to be logical SDutta 22/3/2013 
	/*if(paramNames.size() == 0) {

		return true;
		}*/

	paramNames.push_back("error");
	paramNames.push_back("warning");
	
	std::map<std::string, std::string> namesMap=testsRequested[qtestName];
	
	for(std::vector<std::string>::iterator namesItr=paramNames.begin(); namesItr!=paramNames.end(); ++namesItr){
		if(namesMap.find(*namesItr)==namesMap.end()){
			return true;
		}
	}

	return false;
}

bool QTestConfigurationParser::monitorElementTestsMap(){
	
	std::string testON="true";
	std::string testOFF="false";
	
	unsigned int linkTagsNum  = 
	  doc()->getElementsByTagName(qtxml::_toDOMS("LINK"))->getLength();


	for (unsigned int i=0; i<linkTagsNum; ++i){
	
		DOMNode* linkNode = 
		  doc()->getElementsByTagName(qtxml::_toDOMS("LINK"))->item(i);
		///Get ME name
		if (! linkNode){
			return true;
		}
		DOMElement* linkElement = static_cast<DOMElement *>(linkNode);          
		if (! linkElement){
			return true;		 
		}
		std::string linkName = qtxml::_toString (linkElement->getAttribute (qtxml::_toDOMS ("name"))); 
	
		DOMNodeList *testList = linkElement->getElementsByTagName (qtxml::_toDOMS ("TestName"));
		unsigned int numberOfTests=testList->getLength();
		
		std::vector<std::string> qualityTestList;
		for(unsigned int tt=0; tt<numberOfTests; ++tt){
			DOMElement * testElement = dynamic_cast <DOMElement *> ( testList ->item(tt));		
			if (!testElement ){
				return true;		 
			}
		
			std::string activate = qtxml::_toString (testElement ->getAttribute (qtxml::_toDOMS ("activate"))); 
				
				DOMText *argText = dynamic_cast <DOMText *> (testElement ->getFirstChild());
	   
				if(!std::strcmp(activate.c_str(),testON.c_str()))  {				
					if (!argText){
						return true;
					}else{
						std::string regExpValue = qtxml::_toString (argText->getData());
						qualityTestList.push_back(regExpValue);		
					}		
				}				
				if(!std::strcmp(activate.c_str(),testOFF.c_str())) {
					if (argText) {
						std::string regExpValue = qtxml::_toString (argText->getData());						
						// Create List of QTests to unattach from current ME
					}				
				}
			
		}
	
	
		if(qualityTestList.size()) mapMonitorElementTests[linkName]=qualityTestList;
	}///Loop on linkTagsNum

	
	return false;



}


