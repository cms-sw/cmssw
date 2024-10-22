// -*- C++ -*-
//
// Package:    HcalEmptyEventFilter
// Class:      HcalEmptyEventFilter
//
/**\class HcalEmptyEventFilter HcalEmptyEventFilter.cc filter/HcalEmptyEventFilter/src/HcalEmptyEventFilter.cc

Description: <one line class summary>

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Jeremiah Mans
//         Created:  Tue Jun 4 CET 2012
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>
#include <iostream>

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"
#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"

//
// class declaration
//

class HcalEmptyEventFilter : public edm::global::EDFilter<> {
public:
  explicit HcalEmptyEventFilter(const edm::ParameterSet&);

private:
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------

  edm::EDGetTokenT<FEDRawDataCollection> tok_data_;
};

//
// constructors and destructor
//
HcalEmptyEventFilter::HcalEmptyEventFilter(const edm::ParameterSet& iConfig) {
  //now do what ever initialization is needed

  tok_data_ = consumes<FEDRawDataCollection>(iConfig.getParameter<edm::InputTag>("InputLabel"));
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool HcalEmptyEventFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;

  edm::Handle<FEDRawDataCollection> rawdata;
  iEvent.getByToken(tok_data_, rawdata);

  bool haveEmpty = false;

  for (int i = FEDNumbering::MINHCALFEDID; !haveEmpty && i <= FEDNumbering::MAXHCALFEDID; i++) {
    const FEDRawData& fedData = rawdata->FEDData(i);

    if (fedData.size() < 16)
      continue;

    // get the DCC header
    const HcalDCCHeader* dccHeader = (const HcalDCCHeader*)(fedData.data());

    // walk through the HTR data...
    HcalHTRData htr;

    for (int spigot = 0; spigot < HcalDCCHeader::SPIGOT_COUNT && !haveEmpty; spigot++) {
      if (!dccHeader->getSpigotPresent(spigot))
        continue;

      int retval = dccHeader->getSpigotData(spigot, htr, fedData.size());

      if (retval != 0)
        continue;  // format error is not empty event

      if (htr.isEmptyEvent())
        haveEmpty = true;
    }
  }
  return haveEmpty;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalEmptyEventFilter);
