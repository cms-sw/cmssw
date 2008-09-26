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
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"


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
  SiStripDetInfoFileReader * fr=edm::Service<SiStripDetInfoFileReader>().operator->();
  std::vector<uint32_t> TkDetIdList,fullTkDetIdList=fr->getAllDetIds();

  SiStripSubStructure siStripSubStructure;

  //extract  vector of module in the layer
  siStripSubStructure.getTOBDetectors(fullTkDetIdList,TkDetIdList,0,0,0);
  
  float value;
  for(size_t i=0;i<TkDetIdList.size();++i){
    value = TkDetIdList[i]%1000000;
    //    tkhisto->fill(TkDetIdList[i],value);
    tkhisto->setBinContent(TkDetIdList[i],value);
  }



  siStripSubStructure.getTIBDetectors(fullTkDetIdList,TkDetIdList,0,0,0,0);  
  for(size_t i=0;i<TkDetIdList.size();++i){
    value = TkDetIdList[i]%1000000;
    //    tkhisto->fill(TkDetIdList[i],value);
    tkhisto->setBinContent(TkDetIdList[i],value);
  }
}


//define this as a plug-in
DEFINE_FWK_MODULE(testTkHistoMap);
