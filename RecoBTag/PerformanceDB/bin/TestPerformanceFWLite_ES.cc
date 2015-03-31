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

int main(int argc, char ** argv)
{
  AutoLibraryLoader::enable();
  
  std::cout << "Test!!!" << std::endl << std::endl;
  TFile f("performance_ssvm.root","READ");
      
  fwlite::EventSetup es(&f);

  if ( es.exists("BTagPerformanceRecord") ) {
    std::cout << "Got the right tree" << std::endl;
  } else {
    std::cout << "Can't find tree" << std::endl;
  }

  fwlite::RecordID testRecID = es.recordID("BTagPerformanceRecord");

  int index = 1001;
  es.syncTo(edm::EventID(index,0,0),edm::Timestamp());

      
  std::cout << "Got record ID " << testRecID << es.get(testRecID).startSyncValue().eventID()<<std::endl;

  fwlite::ESHandle< PerformancePayload > plHandle;
  es.get(testRecID).get(plHandle,"MCPfTCHEMb");
  fwlite::ESHandle< PerformanceWorkingPoint > wpHandle;
  es.get(testRecID).get(wpHandle,"MCPfTCHEMb");

  if ( plHandle.isValid() && wpHandle.isValid() ) {
    BtagPerformance perf(*plHandle, *wpHandle);

    std::cout << "Values: "<<
      PerformanceResult::BTAGNBEFF<<" " <<
      PerformanceResult::MUERR<<" " <<
      std::endl;

    // check beff, berr for eta=.6, et=55;
    BinningPointByMap p;

//     std::cout <<" My Performance Object is indeed a "<<typeid(perf).name()<<std::endl;

    std::cout <<" test eta=0.6, et=55"<<std::endl;


    p.insert(BinningVariables::JetAbsEta,0.6);
    p.insert(BinningVariables::JetEt,55);
    std::cout <<" nbeff/nberr ?"<<perf.isResultOk(PerformanceResult::BTAGNBEFF,p)<<"/"<<perf.isResultOk(PerformanceResult::BTAGNBERR,p)<<std::endl;
    std::cout <<" beff/berr ?"<<perf.isResultOk(PerformanceResult::BTAGBEFF,p)<<"/"<<perf.isResultOk(PerformanceResult::BTAGBERR,p)<<std::endl;
    std::cout <<" beff/berr ="<<perf.getResult(PerformanceResult::BTAGBEFF,p)<<"/"<<perf.getResult(PerformanceResult::BTAGBERR,p)<<std::endl;
	
    std::cout <<" test eta=1.9, et=33"<<std::endl;
    p.insert(BinningVariables::JetAbsEta,1.9);
    p.insert(BinningVariables::JetEt,33);
    std::cout <<" beff/berr ?"<<perf.isResultOk(PerformanceResult::BTAGBEFF,p)<<"/"<<perf.isResultOk(PerformanceResult::BTAGBERR,p)<<std::endl;
    std::cout <<" beff/berr ="<<perf.getResult(PerformanceResult::BTAGBEFF,p)<<"/"<<perf.getResult(PerformanceResult::BTAGBERR,p)<<std::endl;
	
    std::cout <<" The WP is defined by a cut at "<<perf.workingPoint().cut()<<std::endl;
    std::cout <<" Discriminant is "<<perf.workingPoint().discriminantName()<<std::endl;
	
    std::cout <<" now I ask for a calibration but I do not set eta in the binning point ---> should return all not available "<<std::endl;
    p.reset();
    p.insert(BinningVariables::JetNTracks,3);
    p.insert(BinningVariables::JetEt,55);
    std::cout <<" beff/berr ?"<<perf.isResultOk(PerformanceResult::BTAGBEFF,p)<<"/"<<perf.isResultOk(PerformanceResult::BTAGBERR,p)<<std::endl;
    std::cout <<" beff/berr ="<<perf.getResult(PerformanceResult::BTAGBEFF,p)<<"/"<<perf.getResult(PerformanceResult::BTAGBERR,p)<<std::endl;
	
    //  std::cout <<" now I ask for a calibration which is not present ---> should throw an exception "<<std::endl;
	
    //  edm::ESHandle<BtagPerformance> perfH2;
    //  iSetup.get<BTagPerformanceRecord>().get("TrackCountingHighEff_tight",perfH2);
	
	
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
