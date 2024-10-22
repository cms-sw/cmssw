// -*- C++ -*-
//
// Package:    EcalFEDWithCRCErrorProducer
// Class:      EcalFEDWithCRCErrorProducer
//
/**\class EcalFEDWithCRCErrorProducer EcalFEDWithCRCErrorProducer.cc filter/EcalFEDWithCRCErrorProducer/src/EcalFEDWithCRCErrorProducer.cc

Description: <one line class summary>

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Giovanni FRANZONI
//         Created:  Tue Jan 22 13:55:00 CET 2008
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/one/EDProducer.h"

#include <string>
#include <iostream>
#include <vector>
#include <iomanip>

#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>
#include <DataFormats/FEDRawData/interface/FEDNumbering.h>
//
// class declaration
//

class EcalFEDWithCRCErrorProducer : public edm::one::EDProducer<> {
public:
  explicit EcalFEDWithCRCErrorProducer(const edm::ParameterSet&);
  ~EcalFEDWithCRCErrorProducer() override;

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------

  edm::EDGetTokenT<FEDRawDataCollection> DataToken_;
  std::vector<int> fedUnpackList_;
  bool writeAllEcalFEDs_;
};

//
// constructors and destructor
//
EcalFEDWithCRCErrorProducer::EcalFEDWithCRCErrorProducer(const edm::ParameterSet& iConfig) {
  //now do what ever initialization is needed

  DataToken_ = consumes<FEDRawDataCollection>(iConfig.getParameter<edm::InputTag>("InputLabel"));
  fedUnpackList_ = iConfig.getUntrackedParameter<std::vector<int> >("FEDs", std::vector<int>());
  writeAllEcalFEDs_ = iConfig.getUntrackedParameter<bool>("writeAllEcalFED", false);
  if (fedUnpackList_.empty())
    for (int i = FEDNumbering::MINECALFEDID; i <= FEDNumbering::MAXECALFEDID; i++)
      fedUnpackList_.push_back(i);

  produces<FEDRawDataCollection>();
}

EcalFEDWithCRCErrorProducer::~EcalFEDWithCRCErrorProducer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called on each new Event  ------------
void EcalFEDWithCRCErrorProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  const edm::Handle<FEDRawDataCollection>& rawdata = iEvent.getHandle(DataToken_);

  auto producedData = std::make_unique<FEDRawDataCollection>();
  // get fed raw data and SM id

  // loop over FEDS
  for (std::vector<int>::const_iterator i = fedUnpackList_.begin(); i != fedUnpackList_.end(); i++) {
    // get fed raw data and SM id
    const FEDRawData& fedData = rawdata->FEDData(*i);
    int length = fedData.size() / sizeof(uint64_t);

    //    LogDebug("EcalRawToDigi") << "raw data length: " << length ;
    //if data size is not null interpret data
    if (length >= 1) {
      uint64_t* pData = (uint64_t*)(fedData.data());
      //When crc error is found return true
      uint64_t* fedTrailer = pData + (length - 1);
      bool crcError = (*fedTrailer >> 2) & 0x1;
      // this fed has data -- lets copy it
      if (writeAllEcalFEDs_ || crcError) {
        FEDRawData& fedDataProd = producedData->FEDData(*i);
        if (fedDataProd.size() != 0) {
          //                edm::LogVerbatim("EcalTools") << " More than one FEDRawDataCollection with data in FED ";
          //                                              << j << " Skipping the 2nd";
          continue;
        }
        fedDataProd.resize(fedData.size());
        unsigned char* dataProd = fedDataProd.data();
        const unsigned char* data = fedData.data();
        for (unsigned int k = 0; k < fedData.size(); ++k) {
          dataProd[k] = data[k];
        }
      }
    }
  }

  iEvent.put(std::move(producedData));
}

//define this as a plug-in
DEFINE_FWK_MODULE(EcalFEDWithCRCErrorProducer);
