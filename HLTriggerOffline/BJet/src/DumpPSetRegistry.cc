#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

class DumpPSetRegistry : public edm::EDAnalyzer {
public:
  explicit DumpPSetRegistry(const edm::ParameterSet& config);
  virtual ~DumpPSetRegistry();
    
  virtual void beginJob(const edm::EventSetup & setup);
  virtual void analyze(const edm::Event & event, const edm::EventSetup & setup);
  virtual void endJob();
};

DumpPSetRegistry::DumpPSetRegistry(const edm::ParameterSet & config) 
{
}

DumpPSetRegistry::~DumpPSetRegistry() 
{
}

void DumpPSetRegistry::beginJob(const edm::EventSetup & setup) 
{
  const edm::pset::Registry * registry = edm::pset::Registry::instance();
  for (edm::pset::Registry::const_iterator i = registry->begin(); i != registry->end(); ++i)
    std::cout << i->second << std::endl;
}

void DumpPSetRegistry::analyze(const edm::Event & event, const edm::EventSetup & setup)
{
}

void DumpPSetRegistry::endJob()
{
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DumpPSetRegistry);
