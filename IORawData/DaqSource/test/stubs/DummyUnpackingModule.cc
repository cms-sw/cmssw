/** \file
 * 
 * dummy module  for the test of  DaqFileInputService
 *   
 * 
 * $Date: 2012/10/10 20:42:01 $
 * $Revision: 1.9 $
 * \author N. Amapane - S. Argiro'
 *
*/

#include <cppunit/extensions/HelperMacros.h>

#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/Utilities/interface/InputTag.h"

#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>
#include <DataFormats/FEDRawData/interface/FEDHeader.h>
#include <DataFormats/FEDRawData/interface/FEDTrailer.h>
#include <DataFormats/FEDRawData/interface/FEDNumbering.h>

#include <iostream>
#include <sys/time.h>
#include <time.h>


using namespace edm;
using namespace std;

namespace test{

  class DummyUnpackingModule: public EDAnalyzer{
  
  private:
    unsigned int count_;

    edm::InputTag fedRawDataCollectionTag_;

  public:

    DummyUnpackingModule(const ParameterSet& pset):count_(0),
      fedRawDataCollectionTag_(pset.getParameter<edm::InputTag>("fedRawDataCollectionTag")) {
    }
 
    void analyze(const Event & e, const EventSetup& c){
      
      ++count_;
      TimeValue_t time = e.time().value();
      time_t stime = time >> 32;
      struct tm uptm;
      gmtime_r(&stime, &uptm);
      char datestring[256];
      strftime(datestring, sizeof(datestring),"%c",&uptm);
      cout << "event date " <<  datestring << endl;
      Handle<FEDRawDataCollection> rawdata;
      e.getByLabel(fedRawDataCollectionTag_, rawdata);
      for (int i = 0; i<FEDNumbering::lastFEDId(); i++){
	const FEDRawData& data = rawdata->FEDData(i);
	if(size_t size=data.size()) {
	  cout << "FED# " << i << " " << size << endl;
	  FEDHeader header(data.data());
	  CPPUNIT_ASSERT(header.check()==true);
	  FEDTrailer trailer(data.data()+size-8);
	  CPPUNIT_ASSERT(trailer.check()==true);
	  CPPUNIT_ASSERT(trailer.lenght()==(int)data.size()/8);
	}
      }
//       if ( count_==1) {
// 	   CPPUNIT_ASSERT( rawdata->FEDData(619).size()==5560);
//         CPPUNIT_ASSERT( rawdata->FEDData(620).size()==5544);     
//       }  
    }
  };
DEFINE_FWK_MODULE(DummyUnpackingModule);
}

