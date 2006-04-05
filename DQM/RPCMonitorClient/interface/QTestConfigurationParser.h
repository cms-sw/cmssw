#ifndef QTestConfigurationParser_H
#define QTestConfigurationParser_H

#include <xercesc/dom/DOM.hpp>
#include <xercesc/dom/DOMCharacterData.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/util/XMLUni.hpp>
#include <xercesc/util/XMLURL.hpp>

          

#include<iostream>
#include<string>
#include<vector>
#include<map>


using namespace xercesc;

class QTestConfigurationParser{

	public:
	 QTestConfigurationParser(std::string configFile);
	 ~QTestConfigurationParser(){}
	 std::map<std::string, std::string> getParams(DOMElement* qtestElement, std::string test);
	 int instances(){return s_numberOfInstances;}
	 
	 /// QTests list obtained from the xml file
	 std::map<std::string, std::map<std::string, std::string> > testsList(){ return testsRequested;}		
	
	private:
	
	 static int s_numberOfInstances;
	 std::map<std::string, unsigned int>   paramsMap;
	 
	 ///       test type             param name   param value
	 std::map<std::string, std::map<std::string, std::string> > testsRequested;	 


};


#endif
