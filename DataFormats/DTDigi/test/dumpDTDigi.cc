#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <DataFormats/DTDigi/interface/DTDigiCollection.h>


#include <iostream>

using namespace edm;
using namespace std;

namespace test{
  class DumpFEDRawDataProduct: public EDAnalyzer{
  
  public:
    DumpFEDRawDataProduct(const ParameterSet& pset){}
 
    void analyze(const Event & e, const EventSetup& c){
      cout << "--- Run: " << e.id().run()
	   << " Event: " << e.id().event() << endl;
      Handle<DTDigiCollection> dtDigis;
      e.getByLabel("dtunpacker", dtDigis);

      DTDigiCollection::DigiRangeIterator detUnitIt;
      for (detUnitIt=digiCollection.begin();
	   detUnitIt!=digiCollection.end();
	   ++detUnitIt){

	const DTLayerId& id = (*detUnitIt).first;
	const DTDigiCollection::Range& range = (*detUnitIt).second;

	// We have inserted digis for only one DetUnit...
	CPPUNIT_ASSERT(id==layer);

	// Loop over the digis of this DetUnit
	for (DTDigiCollection::const_iterator digiIt = range.first;
	     digiIt!=range.second;
	     ++digiIt){


	  CPPUNIT_ASSERT((*digiIt).wire()==1);
	  CPPUNIT_ASSERT((*digiIt).number()==4);
	  CPPUNIT_ASSERT((*digiIt).countsTDC()==5);


	}// for digis in layer
      }// for layers
    }
    
  };
DEFINE_FWK_MODULE(DumpFEDRawDataProduct)      
}
    
