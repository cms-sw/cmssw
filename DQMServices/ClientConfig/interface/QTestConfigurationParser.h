#ifndef QTestConfigurationParser_H
#define QTestConfigurationParser_H

/** \class QTestConfigurationParser
 * *
 *  Parses the xml file with the configuration of quality tests
 *  and the map between quality tests and MonitorElement
 * 
 *  $Date: 2008/05/14 12:38:46 $
 *  $Revision: 1.4 $
 *  \author Ilaria Segoni
  */


#include "DQMServices/ClientConfig/interface/DQMParserBase.h"
       
#include<iostream>
#include<string>
#include<vector>
#include<map>

class QTestParameterNames;

class QTestConfigurationParser : public DQMParserBase {

 public:
	 ///Creator
	 QTestConfigurationParser();
	 ///Destructor
	 ~QTestConfigurationParser();
	 ///Methor that parses the xml file configFile, returns false if no errors are encountered
	 bool parseQTestsConfiguration();
	 /// Returns the Quality Tests list with their parameters obtained from the xml file
	 std::map<std::string, std::map<std::string, std::string> > testsList() const { return testsRequested;}		
	 /// Returns the map between the MonitoElemnt and the list of tests requested for it
	 std::map<std::string, std::vector<std::string> > meToTestsList() const { return mapMonitorElementTests;}		
	
 private:	 
	 bool qtestsConfig();
	 bool monitorElementTestsMap();
	 std::map<std::string, std::string> getParams(xercesc::DOMElement* qtestElement, std::string test);
	 int instances(){return s_numberOfInstances;}
	 bool checkParameters(std::string qtestName, std::string qtestType);
	 
	 
	 
 private:	 
	 static int s_numberOfInstances;
	 	 
	 std::map<std::string, std::map<std::string, std::string> > testsRequested;	 
	 std::map<std::string, std::vector<std::string> >   mapMonitorElementTests;
	 std::vector<std::string> testsToDisable;
	 
	 QTestParameterNames * qtestParamNames;


};


#endif
