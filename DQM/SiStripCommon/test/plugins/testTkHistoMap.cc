// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQM/SiStripCommon/interface/TkHistoMap.h"


#include <math.h>
#include <vector>
#include <sstream>
using std::cout; using std::endl; using std::string;


//
// class declaration
//

class testTkHistoMap : public edm::EDAnalyzer {
public:
  explicit testTkHistoMap ( const edm::ParameterSet& );
  ~testTkHistoMap ();
   
  virtual void analyze( const edm::Event&, const edm::EventSetup& );

  virtual void endJob(void);

private:
    
  TkHistoMap* tkhisto;
    
};
//
testTkHistoMap::testTkHistoMap ( const edm::ParameterSet& iConfig )
{
  tkhisto=new TkHistoMap("pippo","pluto");
}


testTkHistoMap::~testTkHistoMap()
{
   
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  
}

void testTkHistoMap::endJob(void)
{
  edm::Service<DQMStore>().operator->()->save("test.root");  
}


//
// member functions
//

// // ------------ method called to produce the data  ------------
void testTkHistoMap::analyze(const edm::Event& iEvent, 
				     const edm::EventSetup& iSetup )
{   

  uint32_t det=436260900;
  float val=60900;
  tkhisto->setBinContent(det,val);
  det=436260904;
  val=60904;
  //  tkhisto->setBinContent(det,val);
  tkhisto->fill(det,val);
}


//define this as a plug-in
DEFINE_FWK_MODULE(testTkHistoMap);
