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

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"

#include "DataFormats/OnlineMetaData/interface/CTPPSRecord.h"
#include "DataFormats/OnlineMetaData/interface/DCSRecord.h"
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
  reco::BeamSpot getBeamSpot(const online::BeamSpot_v1&) const;

  edm::EDGetTokenT<FEDRawDataCollection> dataToken_;
};

OnlineMetaDataRawToDigi::OnlineMetaDataRawToDigi(const edm::ParameterSet& iConfig) {
  edm::InputTag dataLabel = iConfig.getParameter<edm::InputTag>("onlineMetaDataInputLabel");
  dataToken_ = consumes<FEDRawDataCollection>(dataLabel);

  produces<CTPPSRecord>();
  produces<DCSRecord>();
  produces<OnlineLuminosityRecord>();
  produces<reco::BeamSpot>();
}

OnlineMetaDataRawToDigi::~OnlineMetaDataRawToDigi() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void OnlineMetaDataRawToDigi::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  edm::Handle<FEDRawDataCollection> rawdata;
  iEvent.getByToken(dataToken_, rawdata);

  DCSRecord dcsRecord;
  CTPPSRecord ctppsRecord;
  OnlineLuminosityRecord onlineLuminosityRecord;
  reco::BeamSpot onlineBeamSpot;

  if (rawdata.isValid()) {
    const FEDRawData& onlineMetaDataRaw = rawdata->FEDData(FEDNumbering::MINMetaDataSoftFEDID);
    const unsigned char* payload = onlineMetaDataRaw.data() + FEDHeader::length;

    if (onlineMetaDataRaw.size() >= FEDHeader::length + sizeof(uint8_t)) {
      const uint8_t version = *(reinterpret_cast<uint8_t const*>(payload));
      if (version == 1 && onlineMetaDataRaw.size() >= FEDHeader::length + sizeof(online::Data_v1)) {
        online::Data_v1 const* onlineMetaData = reinterpret_cast<online::Data_v1 const*>(payload);
        dcsRecord = DCSRecord(onlineMetaData->dcs);
        onlineLuminosityRecord = OnlineLuminosityRecord(onlineMetaData->luminosity);
        onlineBeamSpot = getBeamSpot(onlineMetaData->beamSpot);
      } else if (version == 2 && onlineMetaDataRaw.size() >= FEDHeader::length + sizeof(online::Data_v2)) {
        online::Data_v2 const* onlineMetaData = reinterpret_cast<online::Data_v2 const*>(payload);
        ctppsRecord = CTPPSRecord(onlineMetaData->ctpps);
        dcsRecord = DCSRecord(onlineMetaData->dcs);
        onlineLuminosityRecord = OnlineLuminosityRecord(onlineMetaData->luminosity);
        onlineBeamSpot = getBeamSpot(onlineMetaData->beamSpot);
      }
    }
  }

  iEvent.put(std::make_unique<CTPPSRecord>(ctppsRecord));
  iEvent.put(std::make_unique<DCSRecord>(dcsRecord));
  iEvent.put(std::make_unique<OnlineLuminosityRecord>(onlineLuminosityRecord));
  iEvent.put(std::make_unique<reco::BeamSpot>(onlineBeamSpot));
}

reco::BeamSpot OnlineMetaDataRawToDigi::getBeamSpot(const online::BeamSpot_v1& beamSpot) const {
  reco::BeamSpot::Point point(beamSpot.x, beamSpot.y, beamSpot.z);

  reco::BeamSpot::CovarianceMatrix matrix;
  matrix(0, 0) = beamSpot.errX * beamSpot.errX;
  matrix(1, 1) = beamSpot.errY * beamSpot.errY;
  matrix(2, 2) = beamSpot.errZ * beamSpot.errZ;
  matrix(3, 3) = beamSpot.errSigmaZ * beamSpot.errSigmaZ;
  matrix(4, 4) = beamSpot.errDxdz * beamSpot.errDxdz;
  matrix(5, 5) = beamSpot.errDydz * beamSpot.errDydz;
  matrix(6, 6) = beamSpot.errWidthX * beamSpot.errWidthX;
  // Note: errWidthY is not part of the CovarianceMatrix

  reco::BeamSpot bs(
      point, beamSpot.sigmaZ, beamSpot.dxdz, beamSpot.dydz, beamSpot.widthX, matrix, reco::BeamSpot::BeamType::LHC);

  bs.setBeamWidthY(beamSpot.widthY);

  return bs;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void OnlineMetaDataRawToDigi::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("onlineMetaDataInputLabel", edm::InputTag("rawDataCollector"));
  descriptions.add("onlineMetaDataRawToDigi", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(OnlineMetaDataRawToDigi);
