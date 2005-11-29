/** \file
 * 
 * 
 * $Date: 2005/10/06 18:23:47 $
 * $Revision: 1.4 $
 * \author N. Amapane - S. Argiro'
 *
*/

#include <cppunit/extensions/HelperMacros.h>

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
      Handle<FEDRawDataCollection> rawdata;
      e.getByLabel("DaqRawData", rawdata);
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

