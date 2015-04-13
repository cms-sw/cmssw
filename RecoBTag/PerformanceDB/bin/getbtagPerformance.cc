#include <vector>
#include <iostream>

#include "TFile.h"
#include "TTree.h"
#include "FWCore/FWLite/interface/AutoLibraryLoader.h"


#include "PhysicsTools/CondLiteIO/interface/RecordWriter.h"
#include "DataFormats/FWLite/interface/Record.h"
#include "DataFormats/FWLite/interface/EventSetup.h"
#include "DataFormats/FWLite/interface/ESHandle.h"
#include "CondFormats/PhysicsToolsObjects/interface/BinningPointByMap.h"
#include "RecoBTag/PerformanceDB/interface/BtagPerformance.h"
#include "PhysicsTools/FWLite/interface/CommandLineParser.h"

using namespace std;
using optutl::CommandLineParser;

int main(int argc, char ** argv)
{
  // load fwlite libraries
  AutoLibraryLoader::enable();
  // command line options
  optutl::CommandLineParser parser ("get performance");
  
  parser.addOption ("rootFile",   CommandLineParser::kString, 
		    "root filename");
  parser.addOption ("payload", CommandLineParser::kString,
		    "for example MISTAGSSVHEM, MCPfTCHEMb", "MISTAGSSVHEM");
  parser.addOption ("flavor", CommandLineParser::kString,
                    "for example b, c, or l", "b");
  parser.addOption ("type", CommandLineParser::kString,
                    "for example eff or SF", "eff");
  parser.addOption ("pt", CommandLineParser::kDouble,
		    "jet pt");
  parser.addOption ("eta", CommandLineParser::kDouble,
                    "jet eta");

  // Parse the command line arguments
  parser.parseArguments (argc, argv);

  if (argc<3) { 
    parser.help();
  }
  
  string inputFile = parser.stringValue("rootFile");
  string payload   = parser.stringValue("payload");
  string flavor = parser.stringValue("flavor");
  string type = parser.stringValue("type");
  double ajetpt = parser.doubleValue("pt");
  double ajeteta = parser.doubleValue("eta");

  TFile f(inputFile.c_str(),"READ");
      
  fwlite::EventSetup es(&f);

  if ( ! es.exists("BTagPerformanceRecord") ) {
    cout << "Can't find tree" << endl;
    parser.help();
  }

  fwlite::RecordID testRecID = es.recordID("BTagPerformanceRecord");

  int index = 1001;

  es.syncTo(edm::EventID(index,0,0),edm::Timestamp());
      
  fwlite::ESHandle< PerformancePayload > plHandle;
  es.get(testRecID).get(plHandle, payload.c_str() ); // MCPfTCHEMb
  fwlite::ESHandle< PerformanceWorkingPoint > wpHandle;
  es.get(testRecID).get(wpHandle, payload.c_str() );

  if ( plHandle.isValid() && wpHandle.isValid() ) {
    BtagPerformance perf(*plHandle, *wpHandle);

    //std::cout << "Values: "<<
    //  PerformanceResult::BTAGNBEFF<<" " <<
    // PerformanceResult::MUERR<<" " <<
    //  std::endl;

    // check beff, berr for eta=.6, et=55;
    BinningPointByMap p;


    p.insert(BinningVariables::JetAbsEta, ajeteta );
    p.insert(BinningVariables::JetEt, ajetpt );

    //std::cout <<" nbeff/nberr ?"<<perf.isResultOk(PerformanceResult::BTAGNBEFF,p)<<"/"<<perf.isResultOk(PerformanceResult::BTAGNBERR,p)<<std::endl;
    //std::cout <<" beff/berr ?"<<perf.isResultOk(PerformanceResult::BTAGBEFF,p)<<"/"<<perf.isResultOk(PerformanceResult::BTAGBERR,p)<<std::endl;
    if ( type == "eff") {
      if ( flavor == "b" ) cout <<"eff/err = "<<perf.getResult(PerformanceResult::BTAGBEFF,p)<<" / "<<perf.getResult(PerformanceResult::BTAGBERR,p) << endl;
      if ( flavor == "c" ) cout <<"eff/err = "<<perf.getResult(PerformanceResult::BTAGCEFF,p)<<" / "<<perf.getResult(PerformanceResult::BTAGCERR,p) << endl;
      if ( flavor == "l" ) cout <<"eff/err = "<<perf.getResult(PerformanceResult::BTAGLEFF,p)<<" / "<<perf.getResult(PerformanceResult::BTAGLERR,p) << endl;
    } else if ( type == "SF") {
      if ( flavor == "b" ) cout <<"SF/err = "<<perf.getResult(PerformanceResult::BTAGBEFFCORR,p)<<" / "<<perf.getResult(PerformanceResult::BTAGBERRCORR,p) << endl;
      if ( flavor == "c" ) cout <<"SF/err = "<<perf.getResult(PerformanceResult::BTAGCEFFCORR,p)<<" / "<<perf.getResult(PerformanceResult::BTAGCERRCORR,p) << endl;
      if ( flavor == "l" ) cout <<"SF/err = "<<perf.getResult(PerformanceResult::BTAGLEFFCORR,p)<<" / "<<perf.getResult(PerformanceResult::BTAGLERRCORR,p) << endl;
    }
	
	
  } else {
    std::cout << "invalid handle: workingPoint " <<wpHandle.isValid()<<" payload "<<plHandle.isValid()<< std::endl;
    try {
      *wpHandle;
      *plHandle;
    }catch(std::exception& iE) {
      std::cout <<iE.what()<<std::endl;
    }
  }
      
}
