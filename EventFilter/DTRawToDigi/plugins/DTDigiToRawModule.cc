#include "EventFilter/DTRawToDigi/plugins/DTDigiToRawModule.h"
#include "EventFilter/DTRawToDigi/plugins/DTDigiToRaw.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "FWCore/Utilities/interface/CRC16.h"

#include "FWCore/Framework/interface/EventSetup.h"

#include <iostream>

using namespace edm;
using namespace std;

DTDigiToRawModule::DTDigiToRawModule(const edm::ParameterSet& ps) {
  produces<FEDRawDataCollection>();

  dduID = ps.getUntrackedParameter<int>("dduID", 770);
  debug = ps.getUntrackedParameter<bool>("debugMode", false);
  digicoll = consumes<DTDigiCollection>(ps.getParameter<edm::InputTag>("digiColl"));
  mapToken_ = esConsumes<DTReadOutMapping, DTReadOutMappingRcd>();

  useStandardFEDid_ = ps.getUntrackedParameter<bool>("useStandardFEDid", true);
  minFEDid_ = ps.getUntrackedParameter<int>("minFEDid", 770);
  maxFEDid_ = ps.getUntrackedParameter<int>("maxFEDid", 775);

  packer = new DTDigiToRaw(ps);
  if (debug)
    cout << "[DTDigiToRawModule]: constructor" << endl;
}

DTDigiToRawModule::~DTDigiToRawModule() {
  delete packer;
  if (debug)
    cout << "[DTDigiToRawModule]: destructor" << endl;
}

void DTDigiToRawModule::produce(Event& e, const EventSetup& iSetup) {
  auto fed_buffers = std::make_unique<FEDRawDataCollection>();

  // Take digis from the event
  Handle<DTDigiCollection> digis;
  e.getByToken(digicoll, digis);

  // Load DTMap
  edm::ESHandle<DTReadOutMapping> map = iSetup.getHandle(mapToken_);

  // Create the packed data
  int FEDIDmin = 0, FEDIDMax = 0;
  if (useStandardFEDid_) {
    FEDIDmin = FEDNumbering::MINDTFEDID;
    FEDIDMax = FEDNumbering::MAXDTFEDID;
  } else {
    FEDIDmin = minFEDid_;
    FEDIDMax = maxFEDid_;
  }

  for (int id = FEDIDmin; id <= FEDIDMax; ++id) {
    packer->SetdduID(id);
    FEDRawData* rawData = packer->createFedBuffers(*digis, map);

    FEDRawData& fedRawData = fed_buffers->FEDData(id);
    fedRawData = *rawData;
    delete rawData;

    FEDHeader dtFEDHeader(fedRawData.data());
    dtFEDHeader.set(fedRawData.data(), 0, e.id().event(), 0, id);

    FEDTrailer dtFEDTrailer(fedRawData.data() + (fedRawData.size() - 8));
    dtFEDTrailer.set(fedRawData.data() + (fedRawData.size() - 8),
                     fedRawData.size() / 8,
                     evf::compute_crc(fedRawData.data(), fedRawData.size()),
                     0,
                     0);
  }
  // Put the raw data to the event
  e.put(std::move(fed_buffers));
}
