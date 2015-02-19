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

  std::string folder = benchmarkLabel_ ;

  subsystemname_ = "ParticleFlow" ;
  eventInfoFolder_ = subsystemname_ + "/" + folder ;

}


//
// -- BookHistograms
//
void BenchmarkAnalyzer::bookHistograms(DQMStore::IBooker & ibooker,
					    edm::Run const & /* iRun */,
					    edm::EventSetup const & /* iSetup */ )
{
  ibooker.setCurrentFolder(eventInfoFolder_) ;
  cout << "path set to " << eventInfoFolder_ << endl;
}
