//
// Original Author:  Marco Zanetti
//         Created:  Tue Sep  9 15:56:24 CEST 2008


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "EventFilter/FEDInterface/interface/GlobalEventNumber.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include <vector>

class BxNumberFilter : public edm::EDFilter {
public:
  explicit BxNumberFilter(const edm::ParameterSet&);
  ~BxNumberFilter();
  
private:
  virtual void beginJob() ;
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  edm::InputTag inputLabel;
  std::vector<int> goldenBXIds;
  unsigned int range; 
  bool debug;
  

};

BxNumberFilter::BxNumberFilter(const edm::ParameterSet& iConfig) {

  inputLabel = iConfig.getUntrackedParameter<edm::InputTag>("inputLabel",edm::InputTag("source"));
  goldenBXIds = iConfig.getParameter<std::vector<int> >("goldenBXIds");
  range = iConfig.getUntrackedParameter<unsigned int>("range", 1);
  debug = iConfig.getUntrackedParameter<unsigned int>("debug", false);
}


BxNumberFilter::~BxNumberFilter() { }


bool BxNumberFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
   using namespace edm;
   using namespace std;

   bool result = false;

   unsigned int GTEVMId= 812;

   Handle<FEDRawDataCollection> rawdata;
   iEvent.getByLabel(inputLabel, rawdata);  
   const FEDRawData& data = rawdata->FEDData(GTEVMId);
   evf::evtn::evm_board_setformat(data.size());
   // loop over the predefined BX's
   for (vector<int>::const_iterator i = goldenBXIds.begin(); i != goldenBXIds.end(); i++) {

     // Select the BX
     if ( evf::evtn::getfdlbx(data.data()) <= (*i) + range
	  &&
	  evf::evtn::getfdlbx(data.data()) >= (*i) - range ) {
       result = true;
       
       if (debug) {
	 cout << "Event # " << evf::evtn::get(data.data(),true) << endl;
	 cout << "LS # " << evf::evtn::getlbn(data.data()) << endl;
	 cout << "ORBIT # " << evf::evtn::getorbit(data.data()) << endl;
	 cout << "GPS LOW # " << evf::evtn::getgpslow(data.data()) << endl;
	 cout << "GPS HI # " << evf::evtn::getgpshigh(data.data()) << endl;
	 cout << "BX FROM FDL 0-xing # " << evf::evtn::getfdlbx(data.data()) << endl;
       }
       
     } 
   }
   return result;
}

// ------------ method called once each job just before starting event loop  ------------
void  BxNumberFilter::beginJob() {
}

// ------------ method called once each job just after ending the event loop  ------------
void  BxNumberFilter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(BxNumberFilter);
