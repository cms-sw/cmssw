// -*- C++ -*-
//
// Package:    LTCRawToDigi
// Class:      LTCRawToDigi
// 
/**\class LTCRawToDigi LTCRawToDigi.cc EventFilter/LTCRawToDigi/src/LTCRawToDigi.cc

 Description: Unpack FED data to LTC bank. LTCs are FED id 816-823.

 Implementation:
     No comments
*/
//
// Original Author:  Peter Wittich
//         Created:  Tue May  9 07:47:59 CDT 2006
// $Id: LTCRawToDigi.cc,v 1.4 2006/10/27 01:35:24 wmtan Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//FEDRawData 
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
// LTC class
#include "DataFormats/LTCDigi/interface/LTCDigi.h"
//
// class declaration
//

class LTCRawToDigi : public edm::EDProducer {
   public:
      explicit LTCRawToDigi(const edm::ParameterSet&);
      ~LTCRawToDigi();


      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
LTCRawToDigi::LTCRawToDigi(const edm::ParameterSet& iConfig)
{
   //register your products
   produces<LTCDigiCollection>();
}


LTCRawToDigi::~LTCRawToDigi()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
LTCRawToDigi::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  const int LTCFedIDLo = 815;
  const int LTCFedIDHi = 823;

  // Get a handle to the FED data collection
  edm::Handle<FEDRawDataCollection> rawdata;
  iEvent.getByLabel("source" , rawdata);

  // create collection we'll save in the event record
  std::auto_ptr<LTCDigiCollection> pOut(new LTCDigiCollection());

  // Loop over all possible FED's with the appropriate FED ID
  for ( int id = LTCFedIDLo; id <= LTCFedIDHi; ++id ) {
    /// Take a reference to this FED's data
    const FEDRawData & fedData = rawdata->FEDData(id);
    unsigned short int length =  fedData.size();
    if ( ! length ) 
      continue; // bank does not exist
    LTCDigi ltcDigi(fedData.data());
    pOut->push_back(ltcDigi);
  }
  iEvent.put(pOut);
}

//define this as a plug-in
DEFINE_FWK_MODULE(LTCRawToDigi);
