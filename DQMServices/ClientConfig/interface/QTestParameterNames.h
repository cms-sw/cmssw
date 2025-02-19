#ifndef  QTestParameterNames_H
#define  QTestParameterNames_H

/** \class  QTestParameterNames
 * *
 *  Defines name and number of parameters that must be specified in the 
 *  xml configuration file for each quality test besides error and warning thresholds. 
 *  It's used by QTestConfigurationPerser
 *  to check that all necessary parameters are defined.
 * 
 *  $Date: 2006/05/09 21:28:24 $
 *  $Revision: 1.1 $
 *  \author Ilaria Segoni
  */

#include<map>
#include<string>
#include<vector>

struct QTestParameterNames{

  public:
  	///Constructor
	QTestParameterNames();
	///Destructor
	~QTestParameterNames(){}
	///returns the list of parameters used by the test of a given type (the string theTestType 
	///must be one of the names defined in DQMServices/ClientConfig/interface/DQMQualityTestsConfiguration.h
	std::vector<std::string> getTestParamNames(std::string theTestType);

  private:
	void constructMap(std::string testType,
		std::string param1="undefined",std::string param2="undefined",std::string param3="undefined",
		std::string param4="undefined",std::string param5="undefined",
		std::string param6="undefined",std::string param7="undefined"
		,std::string param8="undefined");

  private:
	std::map<std::string, std::vector<std::string> > configurationMap;



};


#endif
