#ifndef  QTestParameterNames_H
#define  QTestParameterNames_H

/** \class  QTestParameterNames
 * *
 *  Parses the xml file with the configuration of quality tests
 *  and the map between quality tests and MonitorElement
 * 
 *  $Date: 2006/04/05 15:43:45 $
 *  $Revision: 1.2 $
 *  \author Ilaria Segoni
  */

#include<map>
#include<string>
#include<vector>

struct QTestParameterNames{

  public:
	QTestParameterNames();
	~QTestParameterNames(){}
	///returns the list of parameters used by the test of a given type (the string used
	///must be one of the names defined in DQM/RPCMonitorClient/interface/DQMQualityTestsConfiguration.h
	std::vector<std::string> getTestParamNames(std::string theTestType);
  private:

  
	void constructMap(std::string testType,
	std::string param1="undefined",std::string param2="undefined",std::string param3="undefined",
	std::string param4="undefined",std::string param5="undefined",
	std::string param6="undefined",std::string param7="undefined"
	,std::string param8="undefined");
	std::map<std::string, std::vector<std::string> > configurationMap;



};


#endif
