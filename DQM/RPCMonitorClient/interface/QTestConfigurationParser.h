#ifndef QTestConfigurationParser_H
#define QTestConfigurationParser_H

/** \class QTestConfigurationParser
 * *
 *  Parses the xml file with the configuration of quality tests
 *  and the map between quality tests and MonitorElement
 * 
 *  $Date: 2006/04/05 15:43:45 $
 *  $Revision: 1.2 $
 *  \author Ilaria Segoni
  */

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

class QTestParameterNames;


using namespace xercesc;

class QTestConfigurationParser{

	public:
	 ///Creator
	 QTestConfigurationParser();
	 ///Destructor
	 ~QTestConfigurationParser();
	 ///Methor that parses the xml file configFile, returns false if no errors are encountered
	 bool parseQTestsConfiguration(std::string configFile);
	 /// Returns the Quality Tests list with their parameters obtained from the xml file
	 std::map<std::string, std::map<std::string, std::string> > testsList() const { return testsRequested;}		
	 /// Returns the map between the MonitoElemnt and the list of tests requested for it
	 std::map<std::string, std::vector<std::string> > meToTestsList() const { return mapMonitorElementTests;}		
	
	private:
	 
	 bool qtestsConfig(DOMDocument* doc);
	 bool monitorElementTestsMap(DOMDocument* doc);
	 std::map<std::string, std::string> getParams(DOMElement* qtestElement, std::string test);
	 int instances(){return s_numberOfInstances;}
	 bool checkParameters(std::string testTypeString);
	 
	 static int s_numberOfInstances;
	 std::map<std::string, unsigned int>   paramsMap;
	 std::vector<std::string>   paramNames;
	 
	 
	 std::map<std::string, std::map<std::string, std::string> > testsRequested;	 
	 std::map<std::string, std::vector<std::string> >   mapMonitorElementTests;
	 
	 QTestParameterNames * qtestParamNames;


};


#endif
