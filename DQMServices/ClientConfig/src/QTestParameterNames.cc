/** \file
 *
 *  Implementation of QTestParameterNames 
 *
 *  $Date: 2006/05/04 10:27:29 $
 *  $Revision: 1.2 $
 *  \author Ilaria Segoni
 */
 
#include<iostream>

#include "DQMServices/ClientConfig/interface/QTestParameterNames.h"
#include "DQMServices/ClientConfig/interface/QTestNames.h"

QTestParameterNames::QTestParameterNames(){
	
	this->constructMap(dqm::qtest_config::XRangeContent, "xmin", "xmax");
	this->constructMap(dqm::qtest_config::YRangeContent, "ymin", "ymax");
	this->constructMap(dqm::qtest_config::DeadChannel,   "threshold");
	this->constructMap(dqm::qtest_config::NoisyChannel,  "tolerance", "neighbours");
	this->constructMap(dqm::qtest_config::MeanInExpectedValue, "mean","useRMS","useSigma","useRange","xmin","xmax");

}
  



void QTestParameterNames::constructMap(std::string testType, std::string param1,std::string param2,std::string param3,
	     std::string param4,std::string param5,std::string param6,std::string param7,std::string param8){
	
	
	std::vector<std::string> paramNames;
	if(std::strcmp(param1.c_str(),"undefined")) paramNames.push_back(param1);       
	if(std::strcmp(param2.c_str(),"undefined")) paramNames.push_back(param2);       
	if(std::strcmp(param3.c_str(),"undefined")) paramNames.push_back(param3);       
	if(std::strcmp(param4.c_str(),"undefined")) paramNames.push_back(param4);       
	if(std::strcmp(param5.c_str(),"undefined")) paramNames.push_back(param5);       
	if(std::strcmp(param6.c_str(),"undefined")) paramNames.push_back(param6);       
	if(std::strcmp(param7.c_str(),"undefined")) paramNames.push_back(param7);       
	if(std::strcmp(param8.c_str(),"undefined")) paramNames.push_back(param8);       
 	
	configurationMap[testType]=paramNames;
	
}



std::vector<std::string> QTestParameterNames::getTestParamNames(std::string theTestType){
	
	if(configurationMap.find(theTestType) != configurationMap.end()) {
		return configurationMap[theTestType];
	}else{
		std::vector<std::string> empty;
		return empty;	
	}
}
