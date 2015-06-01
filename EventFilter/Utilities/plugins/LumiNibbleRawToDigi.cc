// -*- C++ -*-
//
// Package:    ProdTutorial/LumiNibbleRawToDigi
// Class:      LumiNibbleRawToDigi
// 
/**\class LumiNibbleRawToDigi LumiNibbleRawToDigi.cc ProdTutorial/LumiNibbleRawToDigi/plugins/LumiNibbleRawToDigi.cc

 Description:  Producer to unpack lumi nibble from TCDS 

*/
//
// Original Author:  Chris Palmer
//         Created:  Thu, 28 May 2015 19:54:56 GMT
//
//


// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "EventFilter/FEDInterface/interface/GlobalEventNumber.h"
#include "EventFilter/FEDInterface/interface/FED1024.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"


using namespace std;

//
// class declaration
//

class LumiNibbleRawToDigi : public edm::EDProducer {
   public:
      explicit LumiNibbleRawToDigi(const edm::ParameterSet&);
      ~LumiNibbleRawToDigi();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginJob() override;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;
      
      int nibble;
      edm::EDGetTokenT<FEDRawDataCollection> dataToken_;
      
};

LumiNibbleRawToDigi::LumiNibbleRawToDigi(const edm::ParameterSet& iConfig)
{
    edm::InputTag dataLabel = iConfig.getParameter<edm::InputTag>("InputLabel");
    dataToken_=consumes<FEDRawDataCollection>(dataLabel);
    produces<int>( "nibble" ).setBranchAlias( "nibble");
}


LumiNibbleRawToDigi::~LumiNibbleRawToDigi()
{
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
LumiNibbleRawToDigi::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    using namespace edm;

    edm::Handle<FEDRawDataCollection> rawdata;  
    iEvent.getByToken(dataToken_,rawdata);

    if( rawdata.isValid() ) {
        const FEDRawData& tcdsData = rawdata->FEDData(FEDNumbering::MINTCDSuTCAFEDID);
        evf::evtn::TCDSRecord tcdsRecord(tcdsData.data());
        nibble = (int)tcdsRecord.getHeader().getData().header.nibble;
        //std::cout<<"nibble is "<<nibble<<std::endl;
    } else {
        nibble=-1;
    }

    std::unique_ptr<int> pOut(new int(nibble));
    iEvent.put( std::move(pOut), "nibble" );
}

// ------------ method called once each job just before starting event loop  ------------
void 
LumiNibbleRawToDigi::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
LumiNibbleRawToDigi::endJob() {
}

 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
LumiNibbleRawToDigi::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(LumiNibbleRawToDigi);
