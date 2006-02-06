/** \file
 * 
 * 
 * $Date: 2005/12/06 09:25:21 $
 * $Revision: 1.3 $
 * \author N. Amapane - S. Argiro'
 *
*/

#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>
#include <DataFormats/FEDRawData/interface/FEDHeader.h>
#include <DataFormats/FEDRawData/interface/FEDTrailer.h>
#include <DataFormats/FEDRawData/interface/FEDNumbering.h>

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
      Handle<FEDRawDataCollection> rawdata;
      e.getByLabel("DaqSource", rawdata);
      for (int i = 0; i<FEDNumbering::lastFEDId(); i++){
	const FEDRawData& data = rawdata->FEDData(i);
	if(size_t size=data.size()) {
	  cout << "FED# " << i << " " << size << endl;
// 	  FEDHeader header(data.data());
// 	  FEDTrailer trailer(data.data()+size-8);
// 	  CPPUNIT_ASSERT(trailer.check()==true);
// 	  CPPUNIT_ASSERT(trailer.lenght()==(int)data.size()/8);
	}
      }
    }
  };
DEFINE_FWK_MODULE(DumpFEDRawDataProduct)
}

