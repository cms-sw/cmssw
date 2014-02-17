/** \file
 *
 *  Implementation of QTestParameterNames 
 *
 *  $Date: 2012/08/07 04:24:27 $
 *  $Revision: 1.13 $
 *  \author Ilaria Segoni
 */
 
#include <iostream>
#include <cstring>
#include "DQMServices/ClientConfig/interface/QTestParameterNames.h"
#include "DQMServices/Core/interface/QTest.h"

QTestParameterNames::QTestParameterNames(){
	
	this->constructMap(ContentsXRangeROOT::getAlgoName(), "xmin", "xmax");
	this->constructMap(ContentsYRangeROOT::getAlgoName(), "ymin", "ymax","useEmptyBins");
	this->constructMap(DeadChannelROOT::getAlgoName(),   "threshold");
	this->constructMap(NoisyChannelROOT::getAlgoName(),  "tolerance", "neighbours");
	this->constructMap(MeanWithinExpectedROOT::getAlgoName(), "mean","useRMS","useSigma","useRange","xmin","xmax");

	//======================== new quality tests in the parser =====================//
        this->constructMap(Comp2RefEqualHROOT::getAlgoName(), "testparam");
        this->constructMap(Comp2RefChi2ROOT::getAlgoName(), "testparam");
        this->constructMap(Comp2RefKolmogorovROOT::getAlgoName(), "testparam");

//        this->constructMap(MostProbableLandauROOT::getAlgoName(), "xmin", "xmax","normalization", "mostprobable", "sigma");
        this->constructMap(ContentsWithinExpectedROOT::getAlgoName(), "minMean", "maxMean", "minRMS", "maxRMS", "toleranceMean", "minEntries", "useEmptyBins");
        //this->constructMap(ContentsWithinExpectedASROOT::getAlgoName(), "minCont", "maxCont"); 

	this->constructMap(CompareToMedianROOT::getAlgoName(), "MinRel", "MaxRel", "UseEmptyBins", "MinAbs", "MaxAbs");
	this->constructMap(CompareLastFilledBinROOT::getAlgoName(), "AvVal", "MinVal", "MaxVal");

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
