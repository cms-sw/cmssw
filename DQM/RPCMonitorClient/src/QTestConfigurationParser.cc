#include "DQM/RPCMonitorClient/interface/DQMQualityTestsConfiguration.h"
#include "DQM/RPCMonitorClient/interface/QTestConfigurationParser.h"
          

int QTestConfigurationParser::s_numberOfInstances = 0;

inline std::string _toString(const XMLCh *toTranscode){
	std::string tmp(XMLString::transcode(toTranscode));
	return tmp;
}

inline XMLCh*  _toDOMS( std::string temp ){
	XMLCh* buff = XMLString::transcode(temp.c_str());    
	return  buff;
}


QTestConfigurationParser::QTestConfigurationParser(){

	try { 
		std::cout << "Xerces-c initialization Number "
		<< s_numberOfInstances<<std::endl;
		if (s_numberOfInstances==0) 
		XMLPlatformUtils::Initialize();  
	}
	catch (const XMLException& e) {
		std::cout << "Xerces-c error in initialization \n"
		<< "Exception message is:  \n"
		<< _toString(e.getMessage()) <<std::endl;
		///throw and exception here
	}
 
	++s_numberOfInstances;
}


bool QTestConfigurationParser::parseQTestsConfiguration(std::string configFile){

	std::cout<<" Begin Parsing File "<<configFile <<std::endl; 

	XercesDOMParser* parser = new XercesDOMParser;     
	parser->setValidationScheme(XercesDOMParser::Val_Auto);
	parser->setDoNamespaces(false);
	parser->parse(configFile.c_str()); 
	DOMDocument* doc = parser->getDocument();
	assert(doc);

	bool qtErrors= this->qtestsConfig(doc);
	bool meErrors= this->monitorElementTestsMap(doc);
	return (qtErrors||meErrors);

}

bool QTestConfigurationParser::qtestsConfig(DOMDocument* doc){


	paramsMap[dqm::qtest_config::XRangeContent]=dqm::qtest_config::XRangeParams;
	paramsMap[dqm::qtest_config::YRangeContent]=dqm::qtest_config::YRangeParams;
	paramsMap[dqm::qtest_config::DeadChannel]=dqm::qtest_config::DeadChannelParams;
	paramsMap[dqm::qtest_config::NoisyChannel]=dqm::qtest_config::NoisyChannelParams;


	unsigned int qtestTagsNum  = 
 	   doc->getElementsByTagName(_toDOMS("QTEST"))->getLength();


	std::cout<<"Number of Qtests: "<<qtestTagsNum <<std::endl;

	for (unsigned int i=0; i<qtestTagsNum; i++){
		/// Get Node
 		std::cout<<"***\n Test Number: "<<i <<std::endl;
		DOMNode* qtestNode = 
			doc->getElementsByTagName(_toDOMS("QTEST"))->item(i);
	
	
		///Get QTEST name
		if (! qtestNode){
			std::cout<<"Node QTEST does not exist, i="<<i<<std::endl;
			return true;
		}
		DOMElement* qtestElement = static_cast<DOMElement *>(qtestNode);          
		if (! qtestElement){
			std::cout<<"Element QTEST does not exist, i="<<i<<std::endl;
			return true;		 
		}
		std::string qtestName = _toString (qtestElement->getAttribute (_toDOMS ("name"))); 
		std::cout<<"Name of i"<<i<<"-th test: "<< qtestName<<std::endl;
	
	
		///Get Qtest TYPE
		DOMNodeList *typeNodePrefix 
		  = qtestElement->getElementsByTagName (_toDOMS ("TYPE"));
	     	     
		if (typeNodePrefix->getLength () != 1){
			std::cout<<"TYPE is not uniquely defined!"<<std::endl;
			return true;
		}       
	     
		DOMElement *prefixNode = dynamic_cast <DOMElement *> (typeNodePrefix->item (0));
		if (!prefixNode){
			std::cout<<"TYPE does not have value!"<<std::endl;
			return true;
		}
 
	     
		DOMText *prefixText = dynamic_cast <DOMText *> (prefixNode->getFirstChild());
		if (!prefixText){
			std::cout<<"Cannot get TYPE!"<<std::endl;
			return true;
		}
	
		std::string testTypeString = _toString (prefixText->getData ());
		std::cout<<"Test Type= "<< testTypeString<<std::endl;
 	
	
		testsRequested[qtestName]=  this->getParams(qtestElement, testTypeString);	
	
 	} //loop on qtestTagsNum
 
	return false;
 
}

std::map<std::string, std::string> QTestConfigurationParser::getParams(DOMElement* qtestElement, std::string prefix){
	
	std::map<std::string, std::string> paramNamesValues;
	paramNamesValues[dqm::qtest_config::type]=prefix;
	
	unsigned int numberOfParams=paramsMap[prefix];
	DOMNodeList *arguments = qtestElement->getElementsByTagName (_toDOMS ("PARAM"));
		if (arguments->getLength() != numberOfParams){
	 		 std::cout<<"Wrong numbers of parameters: "<<arguments->getLength()<<std::endl;
		}else{	 
			for (unsigned int i=0; i<numberOfParams; i++){
				DOMElement *argNode = dynamic_cast <DOMElement *> ( arguments ->item(i));
				std::string regExp = _toString (argNode->getAttribute (_toDOMS ("name"))); 
				DOMText *argText = dynamic_cast <DOMText *> (argNode->getFirstChild());
				if (!argText){
					std::cout<<"Cannot get value of "<<regExp<<std::endl;
					break;
				}
	   
				std::string regExpValue = _toString (argText->getData());
				paramNamesValues[regExp]=regExpValue;
			}
		}

	return paramNamesValues;

}



bool QTestConfigurationParser::monitorElementTestsMap(DOMDocument* doc){
	
	std::string testON="true";
	
	unsigned int linkTagsNum  = 
 	   doc->getElementsByTagName(_toDOMS("LINK"))->getLength();


	std::cout<<"Number of Links: "<<linkTagsNum <<std::endl;

	for (unsigned int i=0; i<linkTagsNum; i++){
	
		std::cout<<"***\n ME To Test Link Number: "<<i <<std::endl;
		DOMNode* linkNode = 
			doc->getElementsByTagName(_toDOMS("LINK"))->item(i);
		///Get ME name
		if (! linkNode){
			std::cout<<"Node LINK does not exist, i="<<i<<std::endl;
			return true;
		}
		DOMElement* linkElement = static_cast<DOMElement *>(linkNode);          
		if (! linkElement){
			std::cout<<"Element LINK does not exist, i="<<i<<std::endl;
			return true;		 
		}
		std::string linkName = _toString (linkElement->getAttribute (_toDOMS ("name"))); 
		std::cout<<"Name of i"<<i<<"-th ME: "<< linkName<<std::endl;
	
		DOMNodeList *testList = linkElement->getElementsByTagName (_toDOMS ("TestName"));
		unsigned int numberOfTests=testList->getLength();
		
		std::vector<std::string> qualityTestList;
		for(unsigned int tt=0; tt<numberOfTests; ++tt){
			DOMElement * testElement = dynamic_cast <DOMElement *> ( testList ->item(tt));		
			if (!testElement ){
				std::cout<<"Element TestName does not exist, i="<<i<<std::endl;
				return true;		 
			}
		
			std::string activate = _toString (testElement ->getAttribute (_toDOMS ("activate"))); 
			std::cout<<"Test number"<<tt<<"activation: "<< activate<<std::endl;
			if(!std::strcmp(activate.c_str(),testON.c_str())) {
				
				DOMText *argText = dynamic_cast <DOMText *> (testElement ->getFirstChild());
				if (!argText){
					std::cout<<"Cannot get test name"<<std::endl;
					return false;
				}
	   
				std::string regExpValue = _toString (argText->getData());
				qualityTestList.push_back(regExpValue);
				std::cout<<"Test name"<<regExpValue<<std::endl;
			}
		}
	
	
		if(qualityTestList.size()) mapMonitorElementTests[linkName]=qualityTestList;
	}///Loop on linkTagsNum

	
	return false;



}


