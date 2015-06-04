// -*- C++ -*-
//
// Package:    ProdTutorial/TcdsRawToDigi
// Class:      TcdsRawToDigi
// 
/**\class TcdsRawToDigi TcdsRawToDigi.cc ProdTutorial/TcdsRawToDigi/plugins/TcdsRawToDigi.cc

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
#include "FWCore/Framework/interface/stream/EDProducer.h"

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


class TcdsRawToDigi : public edm::stream::EDProducer<> {
   public:
      explicit TcdsRawToDigi(const edm::ParameterSet&);
      ~TcdsRawToDigi();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      
      edm::EDGetTokenT<FEDRawDataCollection> dataToken_;
      
};

TcdsRawToDigi::TcdsRawToDigi(const edm::ParameterSet& iConfig)
{
    edm::InputTag dataLabel = iConfig.getParameter<edm::InputTag>("InputLabel");
    dataToken_=consumes<FEDRawDataCollection>(dataLabel);
    produces<int>( "nibble" ).setBranchAlias( "nibble");
}


TcdsRawToDigi::~TcdsRawToDigi()
{
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void TcdsRawToDigi::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    using namespace edm;

    edm::Handle<FEDRawDataCollection> rawdata;  
    iEvent.getByToken(dataToken_,rawdata);

    int nibble=-99;
    if( rawdata.isValid() ) {
        const FEDRawData& tcdsData = rawdata->FEDData(FEDNumbering::MINTCDSuTCAFEDID);
        if(tcdsData.size()>0){
            evf::evtn::TCDSRecord tcdsRecord(tcdsData.data());
            nibble = (int)tcdsRecord.getHeader().getData().header.nibble;
        } else {
            nibble=-2;
        }
    } else {
        nibble=-1;
    }
    //std::cout<<"nibble is "<<nibble<<std::endl;

    std::unique_ptr<int> pOut(new int(nibble));
    iEvent.put( std::move(pOut), "nibble" );
}

 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
TcdsRawToDigi::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("InputLabel",edm::InputTag("rawDataCollector"));
    descriptions.add("tcdsRawToDigi", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TcdsRawToDigi);
