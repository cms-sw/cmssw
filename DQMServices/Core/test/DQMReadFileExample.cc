// -*- C++ -*-
//
// Package:    DQMServices/CoreROOT
// Class:      DQMReadFileExample
// 
/**\class DQMReadFileExample

Description: Simple example showing how to read MonitorElements from ROOT file

Implementation:
<Notes on implementation>
*/
//
//
//


// system include files
#include <memory>

#include <sstream>
#include <string>
#include <vector>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//
// class declaration
//
class DQMReadFileExample : public edm::EDAnalyzer {
public:
  explicit DQMReadFileExample( const edm::ParameterSet& );
  ~DQMReadFileExample();
  
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
  
  virtual void endJob(void);

private:
  // ----------member data ---------------------------
  
  // back-end interface
  DQMStore * dbe;
  
  // remove all MonitorElements and directories
  void removeAll();
};

//
// constructors and destructor
//
DQMReadFileExample::DQMReadFileExample(const edm::ParameterSet& iConfig ) 
{
  // get hold of back-end interface
  dbe = edm::Service<DQMStore>().operator->();

  std::string filename = iConfig.getUntrackedParameter<std::string>
    ("RootFileName", "test_playback.root");
  dbe->open(filename);
  dbe->showDirStructure();
  removeAll();

  bool overwrite = false; std::string pathname = "Collector/FU0/C1/C2";
  dbe->open(filename, overwrite, pathname);
  dbe->showDirStructure();
  removeAll();

  pathname = "Collector/FU0/C1/C3";
  dbe->open(filename, overwrite, pathname);
  dbe->showDirStructure();
  removeAll();
}



DQMReadFileExample::~DQMReadFileExample()
{
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  
}

// remove all MonitorElements and directories
void DQMReadFileExample::removeAll()
{
  // go to top directory
  dbe->cd();
  // remove MEs at top directory
  dbe->removeContents();
  // remove directory (including subdirectories recursively)
  if(dbe->dirExists("Collector"))
     dbe->rmdir("Collector");
  if(dbe->dirExists("Summary"))
  dbe->rmdir("Summary");
}

void DQMReadFileExample::endJob(void)
{
  dbe->showDirStructure();
  // dbe->save("test.root");  
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void DQMReadFileExample::analyze(const edm::Event& iEvent, 
					 const edm::EventSetup& iSetup )
{

}

// define this as a plug-in
DEFINE_FWK_MODULE(DQMReadFileExample);

