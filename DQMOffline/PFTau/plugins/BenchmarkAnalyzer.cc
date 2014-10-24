#include "DQMOffline/PFTau/plugins/BenchmarkAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <string>
#include <iostream>

using namespace std;

BenchmarkAnalyzer::BenchmarkAnalyzer(const edm::ParameterSet& parameterSet)
{

  inputLabel_      = parameterSet.getParameter<edm::InputTag>("InputCollection");
  benchmarkLabel_  = parameterSet.getParameter<std::string>("BenchmarkLabel"); 


}



void 
BenchmarkAnalyzer::beginJob()
{  
  Benchmark::DQM_ = edm::Service<DQMStore>().operator->();
  if(!Benchmark::DQM_) {
    throw "Please initialize the DQM service in your cfg";
  }

  // part of the following could be put in the base class
  string path = "PFTask/" + benchmarkLabel_ ; 
  Benchmark::DQM_->setCurrentFolder(path.c_str());
  cout<<"path set to "<<path<<endl;
}
