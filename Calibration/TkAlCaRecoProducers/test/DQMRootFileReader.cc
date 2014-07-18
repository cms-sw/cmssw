// -*- C++ -*-
//
// Class:      DQMRootFileReader
// 
/**\class DQMRootFileReader

Description: Simple example showing how to read MonitorElements from a DQM plain ROOT file

Implementation:
<Notes on implementation>
*/
//
//
//


// system include files
#include <string>
#include <vector>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"


using std::cout;
using std::endl;

//
// class declaration
//
class DQMRootFileReader : public edm::EDAnalyzer {
public:
  explicit DQMRootFileReader( const edm::ParameterSet& );
  ~DQMRootFileReader();
  
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
  
  virtual void endJob(void);

private:
  // ----------member data ---------------------------
  
  // back-end interface
  DQMStore * dbe;
  std::string filename;
};

//
// constructors and destructor
//
DQMRootFileReader::DQMRootFileReader(const edm::ParameterSet& iConfig ) {
  // get hold of back-end interface
  dbe = edm::Service<DQMStore>().operator->();

  filename = iConfig.getUntrackedParameter<std::string>("RootFileName", "test_playback.root");
}



DQMRootFileReader::~DQMRootFileReader() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  
}


void DQMRootFileReader::endJob(void)
{
  cout << "Dumping DQMStore dir structure:" << endl;
  dbe->showDirStructure();
  // dbe->save("test.root");  
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void DQMRootFileReader::analyze(const edm::Event& iEvent, 
					 const edm::EventSetup& iSetup )
{
  // NOTE: this is here just because we need it after the beginRun of MEtoEDMCoverter which calls a Reset on all MEs.
  dbe->open(filename, false, "","",DQMStore::OpenRunDirs::StripRunDirs);
  dbe->showDirStructure();

}

// define this as a plug-in
DEFINE_FWK_MODULE(DQMRootFileReader);

