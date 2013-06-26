/** \file
 * 
 * 
 * $Date: 2010/04/26 16:00:37 $
 * $Revision: 1.7 $
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
#include <iomanip>

using namespace edm;
using namespace std;

namespace test{

  class DumpFEDRawDataProduct: public EDAnalyzer{
  private:
    std::set<int> FEDids_;
    std::string label_;
    bool dumpPayload_;
  public:
    DumpFEDRawDataProduct(const ParameterSet& pset){
      std::vector<int> ids;
      label_ = pset.getUntrackedParameter<std::string>("label","source");
      ids=pset.getUntrackedParameter<std::vector<int> >("feds",std::vector<int>());
      dumpPayload_=pset.getUntrackedParameter<bool>("dumpPayload",false);
      for (std::vector<int>::iterator i=ids.begin(); i!=ids.end(); i++) 
	FEDids_.insert(*i);
    }

 
    void analyze(const Event & e, const EventSetup& c){
      cout << "--- Run: " << e.id().run()
	   << " Event: " << e.id().event() << endl;
      Handle<FEDRawDataCollection> rawdata;
      e.getByLabel(label_,rawdata);
      for (int i = 0; i<=FEDNumbering::lastFEDId(); i++){
	const FEDRawData& data = rawdata->FEDData(i);
	size_t size=data.size();

	if (size>0 && (FEDids_.empty() || FEDids_.find(i)!=FEDids_.end())) {
	  cout << "FED# " << setw(4) << i << " " << setw(8) << size << " bytes " ;
	  
 	  FEDHeader header(data.data());
 	  FEDTrailer trailer(data.data()+size-8);

	  cout << " L1Id: " << setw(8) << header.lvl1ID();
	  cout << " BXId: " << setw(4) << header.bxID();
	  cout << endl;
	  
	  if (dumpPayload_) {
	    const uint64_t* payload=(uint64_t*)(data.data());
	    cout << hex << setfill('0');
	    for (unsigned int i=0; i<data.size()/sizeof(uint64_t); i++) {
	      cout << setw(4) << i << "  " << setw(16) << payload[i] << endl;
	    }
	    cout << dec << setfill(' ');
	  }

// 	  CPPUNIT_ASSERT(trailer.check()==true);
// 	  CPPUNIT_ASSERT(trailer.lenght()==(int)data.size()/8);
	}
      }
    }
  };
DEFINE_FWK_MODULE(DumpFEDRawDataProduct);
}

