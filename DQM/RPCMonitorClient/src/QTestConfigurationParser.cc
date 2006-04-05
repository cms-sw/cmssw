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



QTestConfigurationParser::QTestConfigurationParser(std::string configFile){

/// INITIALIZE
 std::cout<<" Begin Parsing"<<std::endl; 
 
 paramsMap[dqm::qtest_config::XRangeContent]=dqm::qtest_config::XRangeParams;
 paramsMap[dqm::qtest_config::YRangeContent]=dqm::qtest_config::YRangeParams;
 paramsMap[dqm::qtest_config::DeadChannel]=dqm::qtest_config::DeadChannelParams;
 paramsMap[dqm::qtest_config::NoisyChannel]=dqm::qtest_config::NoisyChannelParams;
 

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

/// PARSE THE XML FILE
 XercesDOMParser* parser = new XercesDOMParser;     
 parser->setValidationScheme(XercesDOMParser::Val_Auto);
 parser->setDoNamespaces(false);
 parser->parse(configFile.c_str()); 
 DOMDocument* doc = parser->getDocument();
 assert(doc);



/// NAVIGATE THE TREE



unsigned int ruleTagsNum  = 
    doc->getElementsByTagName(_toDOMS("QTEST"))->getLength();


 std::cout<<"Number of Qtests: "<<ruleTagsNum <<std::endl;

 for (unsigned int i=0; i<ruleTagsNum; i++){
	/// Get Node
 	std::cout<<"***\n Test Number: "<<i <<std::endl;
	DOMNode* ruleNode = 
		doc->getElementsByTagName(_toDOMS("QTEST"))->item(i);
	
	
	///Get QTEST name
	if (! ruleNode){
			std::cout<<"Node QTEST does not exist, i="<<i<<std::endl;
			break;
	}
	DOMElement* qtestElement = static_cast<DOMElement *>(ruleNode);          
	if (! qtestElement){
		std::cout<<"Element QTEST does not exist, i="<<i<<std::endl;
		break;		 
	}
	std::string regExp = _toString (qtestElement->getAttribute (_toDOMS ("name"))); 
	std::cout<<"Name of i"<<i<<"-th test: "<< regExp<<std::endl;
	
	
	///Get Qtest TYPE
	DOMNodeList *prefixes 
		 = qtestElement->getElementsByTagName (_toDOMS ("TYPE"));
	     	     
	if (prefixes->getLength () != 1){
		std::cout<<"TYPE is not uniquely defined!"<<std::endl;
		break;
	}       
	     
	DOMElement *prefixNode = dynamic_cast <DOMElement *> (prefixes->item (0));
	if (!prefixNode){
		std::cout<<"TYPE does not have value!"<<std::endl;
		break;
	}
 
	     
	DOMText *prefixText = dynamic_cast <DOMText *> (prefixNode->getFirstChild());
	if (!prefixText){
		std::cout<<"Cannot get TYPE!"<<std::endl;
		break;
	}
	
	std::string prefix = _toString (prefixText->getData ());
	std::cout<<"Test Type= "<<prefix <<std::endl;
 	
	
	testsRequested[regExp]=  this->getParams(qtestElement, prefix);	
	
 } //loop on QTEST nodes
 
 
 
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

