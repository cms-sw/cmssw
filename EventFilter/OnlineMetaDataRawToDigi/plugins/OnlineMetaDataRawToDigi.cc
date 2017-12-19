// -*- C++ -*-
//
// Package:    EventFilter/OnlineMetaDataRawToDigi
// Class:      OnlineMetaDataRawToDigi
//
/**\class OnlineMetaDataRawToDigi OnlineMetaDataRawToDigi.cc EventFilter/OnlineMetaDataRawToDigi/plugins/OnlineMetaDataRawToDigi.cc

 Description:  Producer to unpack event meta-data from soft-FED 1022

*/
//
// Original Author:  Remigius K Mommsen (Fermilab)
//         Created:  Wed, 22 Nov 2017 18:19:56 GMT
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
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"

#include "DataFormats/OnlineMetaData/interface/DCSRecord.h"
#include "DataFormats/OnlineMetaData/interface/OnlineBeamSpotRecord.h"
#include "DataFormats/OnlineMetaData/interface/OnlineLuminosityRecord.h"
#include "DataFormats/OnlineMetaData/interface/OnlineMetaDataRaw.h"


//
// class declaration
//


class OnlineMetaDataRawToDigi : public edm::stream::EDProducer<> {
public:
  explicit OnlineMetaDataRawToDigi(const edm::ParameterSet&);
  ~OnlineMetaDataRawToDigi() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  edm::EDGetTokenT<FEDRawDataCollection> dataToken_;

};

OnlineMetaDataRawToDigi::OnlineMetaDataRawToDigi(const edm::ParameterSet& iConfig)
{
  edm::InputTag dataLabel = iConfig.getParameter<edm::InputTag>("onlineMetaDataInputLabel");
  dataToken_=consumes<FEDRawDataCollection>(dataLabel);

  produces<DCSRecord>("dcsRecord");
  produces<OnlineBeamSpotRecord>("onlineBeamSpotRecord");
  produces<OnlineLuminosityRecord>("onlineLuminosityRecord");
}


OnlineMetaDataRawToDigi::~OnlineMetaDataRawToDigi() {}


//
// member functions
//

// ------------ method called to produce the data  ------------
void OnlineMetaDataRawToDigi::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  edm::Handle<FEDRawDataCollection> rawdata;
  iEvent.getByToken(dataToken_,rawdata);

  DCSRecord dcsRecord;
  OnlineBeamSpotRecord onlineBeamSpotRecord;
  OnlineLuminosityRecord onlineLuminosityRecord;

  if( rawdata.isValid() ) {
    const FEDRawData& onlineMetaDataRaw = rawdata->FEDData(FEDNumbering::MINMetaDataSoftFEDID);

    if ( onlineMetaDataRaw.size() >= sizeof(online::Data_v1) + FEDHeader::length ) {
      online::Data_v1 const* onlineMetaData =
        reinterpret_cast<online::Data_v1 const*>(onlineMetaDataRaw.data() + FEDHeader::length);
      dcsRecord = DCSRecord(onlineMetaData->dcs);
      onlineBeamSpotRecord = OnlineBeamSpotRecord(onlineMetaData->beamSpot);
      onlineLuminosityRecord = OnlineLuminosityRecord(onlineMetaData->luminosity);
    }
  }

  iEvent.put(std::make_unique<DCSRecord>(dcsRecord), "dcsRecord");
  iEvent.put(std::make_unique<OnlineBeamSpotRecord>(onlineBeamSpotRecord), "onlineBeamSpotRecord");
  iEvent.put(std::make_unique<OnlineLuminosityRecord>(onlineLuminosityRecord), "onlineLuminosityRecord");
}


// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void OnlineMetaDataRawToDigi::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("onlineMetaDataInputLabel",edm::InputTag("rawDataCollector"));
  descriptions.add("onlineMetaDataRawToDigi", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(OnlineMetaDataRawToDigi);
