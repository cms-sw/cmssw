#include "DQM/RPCMonitorClient/interface/QTestConfigurationParser.h"
#include "DQM/RPCMonitorClient/interface/QTestNames.h"
#include "DQM/RPCMonitorClient/interface/QTestParameterNames.h"
#include "DQM/RPCMonitorClient/interface/QTestDefineDebug.h"
#include "DQM/RPCMonitorClient/interface/ParserFunctions.h"

#include <stdexcept>         
/** \file
 *
 *  Implementation of QTestConfigurationParser
 *
 *  $Date: 2006/04/24 10:04:31 $
 *  $Revision: 1.2 $
 *  \author Ilaria Segoni
 */

int QTestConfigurationParser::s_numberOfInstances = 0;


QTestConfigurationParser::QTestConfigurationParser(){
        
	qtestParamNames=new QTestParameterNames();

	try { 
		#ifdef QT_MANAGING_DEBUG
		std::cout << "Xerces-c initialization Number "
		<< s_numberOfInstances<<std::endl;
		#endif
		if (s_numberOfInstances==0) 
		XMLPlatformUtils::Initialize();  
	}
	catch (const XMLException& e) {
		#ifdef QT_MANAGING_DEBUG
		std::cout << "Xerces-c error in initialization \n"
		<< "Exception message is:  \n"
		<< qtxml::_toString(e.getMessage()) <<std::endl;
		#endif
		throw(std::runtime_error("Standard pool exception : Fatal Error on pool::TrivialFileCatalog"));
	}
 
	++s_numberOfInstances;
}

QTestConfigurationParser::~QTestConfigurationParser(){
	delete qtestParamNames;
	qtestParamNames = 0;
}

bool QTestConfigurationParser::parseQTestsConfiguration(){
	
	bool qtErrors= this->qtestsConfig();
	bool meErrors= this->monitorElementTestsMap();
	return (qtErrors||meErrors);

}

bool QTestConfigurationParser::qtestsConfig(){
	

	unsigned int qtestTagsNum  = 
 	   doc->getElementsByTagName(qtxml::_toDOMS("QTEST"))->getLength();

	for (unsigned int i=0; i<qtestTagsNum; ++i){
		/// Get Node
		DOMNode* qtestNode = 
			doc->getElementsByTagName(qtxml::_toDOMS("QTEST"))->item(i);
	
	
		///Get QTEST name
		if (! qtestNode){
			#ifdef QT_MANAGING_DEBUG
			std::cout<<"Node QTEST does not exist, i="<<i<<std::endl;
			#endif
			return true;
		}
		DOMElement* qtestElement = static_cast<DOMElement *>(qtestNode);          
		if (! qtestElement){
			#ifdef QT_MANAGING_DEBUG
			std::cout<<"Element QTEST does not exist, i="<<i<<std::endl;
			#endif
			return true;		 
		}
		std::string qtestName = qtxml::_toString (qtestElement->getAttribute (qtxml::_toDOMS ("name"))); 
	
		///Get Qtest TYPE
		DOMNodeList *typeNodePrefix 
		  = qtestElement->getElementsByTagName (qtxml::_toDOMS ("TYPE"));
	     	     
		if (typeNodePrefix->getLength () != 1){
			#ifdef QT_MANAGING_DEBUG
			std::cout<<"TYPE is not uniquely defined!"<<std::endl;
			#endif
			return true;
		}       
	     
		DOMElement *prefixNode = dynamic_cast <DOMElement *> (typeNodePrefix->item (0));
		if (!prefixNode){
			#ifdef QT_MANAGING_DEBUG
			std::cout<<"TYPE does not have value!"<<std::endl;
			#endif
			return true;
		}
 
	     
		DOMText *prefixText = dynamic_cast <DOMText *> (prefixNode->getFirstChild());
		if (!prefixText){	
			#ifdef QT_MANAGING_DEBUG
			std::cout<<"Cannot get TYPE!"<<std::endl;
			#endif
			return true;
		}
	
		std::string qtestType = qtxml::_toString (prefixText->getData ());

		testsRequested[qtestName]=  this->getParams(qtestElement, qtestType);
		
		if( this->checkParameters(qtestName, qtestType)) return true;
	
 	} //loop on qtestTagsNum
 
	return false;
 
}

std::map<std::string, std::string> QTestConfigurationParser::getParams(DOMElement* qtestElement, std::string qtestType){
	
	std::map<std::string, std::string> paramNamesValues;
	paramNamesValues[dqm::qtest_config::type]=qtestType;
	
	DOMNodeList *arguments = qtestElement->getElementsByTagName (qtxml::_toDOMS ("PARAM"));
	
	for (unsigned int i=0; i<arguments->getLength(); ++i){
		DOMElement *argNode = dynamic_cast <DOMElement *> ( arguments ->item(i));
		std::string regExp = qtxml::_toString (argNode->getAttribute (qtxml::_toDOMS ("name"))); 
		DOMText *argText = dynamic_cast <DOMText *> (argNode->getFirstChild());
		if (!argText){
			#ifdef QT_MANAGING_DEBUG
			std::cout<<"Cannot get value of "<<regExp<<std::endl;
			#endif
			break;
		}
	   
		std::string regExpValue = qtxml::_toString (argText->getData());
		paramNamesValues[regExp]=regExpValue;
	}
        
	return paramNamesValues;

}

bool QTestConfigurationParser::checkParameters(std::string qtestName, std::string qtestType){
	
	std::vector<std::string> paramNames=qtestParamNames->getTestParamNames(qtestType);
	if(paramNames.size() == 0) {
		#ifdef QT_MANAGING_DEBUG
		std::cout<<"Parameters for test type "<< qtestType<<" are not defined, please check .xml file"<<std::endl;
		#endif
		return true;
	}
	paramNames.push_back("error");
	paramNames.push_back("warning");
	
	std::map<std::string, std::string> namesMap=testsRequested[qtestName];
	
	for(std::vector<std::string>::iterator namesItr=paramNames.begin(); namesItr!=paramNames.end(); ++namesItr){
		if(namesMap.find(*namesItr)==namesMap.end()){
			#ifdef QT_MANAGING_DEBUG
			std::cout<<"Parameter ``"<<*namesItr<<"'' for test "<<qtestName<<" is not defined"<<std::endl;
			#endif
			return true;
		}
	}

	return false;
}

bool QTestConfigurationParser::monitorElementTestsMap(){
	
	std::string testON="true";
	
	unsigned int linkTagsNum  = 
 	   doc->getElementsByTagName(qtxml::_toDOMS("LINK"))->getLength();


	for (unsigned int i=0; i<linkTagsNum; ++i){
	
		DOMNode* linkNode = 
			doc->getElementsByTagName(qtxml::_toDOMS("LINK"))->item(i);
		///Get ME name
		if (! linkNode){
			#ifdef QT_MANAGING_DEBUG
			std::cout<<"Node LINK does not exist, i="<<i<<std::endl;
			#endif
			return true;
		}
		DOMElement* linkElement = static_cast<DOMElement *>(linkNode);          
		if (! linkElement){
			#ifdef QT_MANAGING_DEBUG
			std::cout<<"Element LINK does not exist, i="<<i<<std::endl;
			#endif
			return true;		 
		}
		std::string linkName = qtxml::_toString (linkElement->getAttribute (qtxml::_toDOMS ("name"))); 
	
		DOMNodeList *testList = linkElement->getElementsByTagName (qtxml::_toDOMS ("TestName"));
		unsigned int numberOfTests=testList->getLength();
		
		std::vector<std::string> qualityTestList;
		for(unsigned int tt=0; tt<numberOfTests; ++tt){
			DOMElement * testElement = dynamic_cast <DOMElement *> ( testList ->item(tt));		
			if (!testElement ){
				#ifdef QT_MANAGING_DEBUG
				std::cout<<"Element TestName does not exist, i="<<i<<std::endl;
				#endif
				return true;		 
			}
		
			std::string activate = qtxml::_toString (testElement ->getAttribute (qtxml::_toDOMS ("activate"))); 
			if(!std::strcmp(activate.c_str(),testON.c_str())) {
				
				DOMText *argText = dynamic_cast <DOMText *> (testElement ->getFirstChild());
				if (!argText){
					#ifdef QT_MANAGING_DEBUG
					std::cout<<"Cannot get test name"<<std::endl;
					#endif
					return true;
				}
	   
				std::string regExpValue = qtxml::_toString (argText->getData());
				qualityTestList.push_back(regExpValue);
			}
		}
	
	
		if(qualityTestList.size()) mapMonitorElementTests[linkName]=qualityTestList;
	}///Loop on linkTagsNum

	
	return false;



}


