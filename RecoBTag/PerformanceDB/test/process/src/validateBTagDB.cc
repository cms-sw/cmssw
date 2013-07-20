// -*- C++ -*-
//
// Package:    validateBTagDB
// Class:      validateBTagDB
// 
/**\class TestOctoberExe TestOctoberExe.cc RecoBTag/TestOctoberExe/src/TestOctoberExe.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Tommaso Boccali
//         Created:  Tue Nov 25 15:50:50 CET 2008
// $Id: validateBTagDB.cc,v 1.2 2013/01/31 17:54:45 msegala Exp $
//
//


// system include files
#include <memory>
#include <iostream>
#include <fstream>
#include <utility>
#include <map>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
//
// class decleration
//

#include "RecoBTag/Records/interface/BTagPerformanceRecord.h"

#include "CondFormats/PhysicsToolsObjects/interface/BinningPointByMap.h"
#include "RecoBTag/PerformanceDB/interface/BtagPerformance.h"
//#include "CondFormats/BTagPerformance/interface/BtagPerformancePayloadFromTableEtaJetEt.h"
//#include "CondFormats/BTagPerformance/interface/BtagPerformancePayloadFromTableEtaJetEtPhi.h"

class validateBTagDB : public edm::EDAnalyzer {
public:
  explicit validateBTagDB(const edm::ParameterSet&);
  ~validateBTagDB();
  
  
private:
    std::string beff,mistag,ceff;
    std::vector<std::string> algoNames;
    std::vector<std::string> fileList;
    virtual void beginJob() ;
    virtual void analyze(const edm::Event&, const edm::EventSetup&);
    virtual void endJob() ;
  
    // ----------member data ---------------------------
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
validateBTagDB::validateBTagDB(const edm::ParameterSet& iConfig)

{
    //now do what ever initialization is needed
    std::cout <<" In the constructor"<<std::endl;
  
    beff =  iConfig.getParameter<std::string>("CalibrationForBEfficiency");
    ceff =  iConfig.getParameter<std::string>("CalibrationForCEfficiency");
    mistag =  iConfig.getParameter<std::string>("CalibrationForMistag");
    algoNames = iConfig.getParameter<std::vector<std::string> >("algoNames");
    fileList = iConfig.getParameter<std::vector<std::string> >("fileList");

}


validateBTagDB::~validateBTagDB()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
validateBTagDB::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    if( fileList.size() < algoNames.size() )
    {
	std::cout << "File list short!" << std::endl;
	exit(1);
    }
    std::cout << "Cut checks" << std::endl;
    for( size_t i = 0; i < algoNames.size(); i++ )
    {
	edm::ESHandle<BtagPerformance> perfRecord;
	iSetup.get<BTagPerformanceRecord>().get(algoNames[i],perfRecord);
	BtagPerformance perfTest = *(perfRecord.product());
	std::cout << algoNames[i] << " " << perfTest.workingPoint().cut() << std::endl;
	std::cout << "Checking against: " << fileList[i] << std::endl;
	std::ifstream inFile(fileList[i].c_str());
	std::ofstream outFile( ("text/" + algoNames[i] + ".txt").c_str() );
	std::ostringstream output;

	int nMeasures;
	int nVars;
	std::string name;
	float workingPoint;
	std::string type;
std::cout << "1" << std::endl;
//Name
	inFile >> name;
	output << name << std::endl;
std::cout << "2" << std::endl;
//WP
	inFile >> workingPoint;
	output << workingPoint << std::endl;
std::cout << "3" << std::endl;
//Payload type
	inFile >> type;
	output << type << std::endl;
std::cout << "4" << std::endl;
//N measurements
	inFile >> nMeasures;
	output << nMeasures << std::endl;
std::cout << "5" << std::endl;
//N Vars
	inFile >> nVars;
	output << nVars << std::endl;
std::cout << "6" << std::endl;
//Measure enums
	std::vector< int > measureList;
	for( int iM = 1; iM <= nMeasures; iM++ )
	{
	    int temp;
	    inFile >> temp;
	    measureList.push_back( temp );
	    output << measureList[ iM - 1 ] << " ";
	}
	output << std::endl;
std::cout << "7" << std::endl;
//Vars enums
	std::vector< int > varList;
	for( int iV = 1; iV <= nVars; iV++ )
	{
	    int temp;
	    inFile >> temp;
	    varList.push_back( temp );
	    output << varList[ iV - 1 ] << " ";
	}
	output << std::endl;
std::cout << "8" << std::endl;
	while( !inFile.eof() )
	{
//	std::vector< std::pair<float,float> > varBins;
	    BinningPointByMap tempMeasure;
	    bool done = false;
	    //std::cout <<"nvars = "<< nVars << std::endl;
	    for( int iV = 1; iV <= nVars; iV++ )
	    {
		float val1, val2;
		inFile >> val1;
		inFile >> val2;
//	    varBins.push_back( std::make_pair( val1, val2 ) );
		output << val1 << " " << val2 << " ";
		tempMeasure.insert((BinningVariables::BinningVariablesType)varList[iV - 1], (val1+val2)/2.0);
	    }
//std::cout << "9" << std::endl;
//Measurement goes here!
	    for( int iM = 1; iM <= nMeasures; iM++ )
	    {
	      if( inFile.peek() == EOF ){ done = true; }
	    }
	    if(done){ continue; }
	    for( int iM = 1; iM <= nMeasures; iM++ )
	    {
		float res1, measured;
		measured = perfTest.getResult( (PerformanceResult::ResultType)measureList[iM -1], tempMeasure );
		inFile >> res1;
		output << measured << " ";
		std::cout << measured << std::endl;
		if( res1 != measured )
		{
		    std::cout << "Measure/result mismatch with" << measured << " " << res1 << std::endl;
//		output << " check! ";
		}
	    }
	    output << std::endl;
	}
//	std::cout << output.str();
	outFile << output.str();
	outFile.close();
	inFile.close();

//    std::cout << output.str() << std::endl;
  }

}


// ------------ method called once each job just before starting event loop  ------------
void 
validateBTagDB::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
validateBTagDB::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(validateBTagDB);
