// -*- C++ -*-
//
// Package:    TMTFilter
// Class:      TMTFilter
//
/**\class TMTFilter TMTFilter.cc EventFilter/L1TRawToDigi/src/TMTFilter.cc

Description: <one line class summary>
Implementation:
<Notes on implementation>
*/
//
// Original Author:  Jim Brooke
//         Created:
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
#include "FWCore/Utilities/interface/InputTag.h"

#include <string>
#include <vector>
#include <iostream>

#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"

//
// class declaration
//

class TMTFilter : public edm::global::EDFilter<> {
public:
  explicit TMTFilter(const edm::ParameterSet&);

private:
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<FEDRawDataCollection> fedData_;

  std::vector<int> mpList_;  // list of MPs to select
};

//
// constructors and destructor
//
TMTFilter::TMTFilter(const edm::ParameterSet& iConfig)
    : mpList_(iConfig.getUntrackedParameter<std::vector<int> >("mpList")) {
  //now do what ever initialization is needed

  fedData_ = consumes<FEDRawDataCollection>(iConfig.getParameter<edm::InputTag>("inputTag"));
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool TMTFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;

  edm::Handle<FEDRawDataCollection> feds;
  iEvent.getByToken(fedData_, feds);

  if (!feds.isValid()) {
    LogError("L1T") << "Cannot unpack: no FEDRawDataCollection found";
    return false;
  }

  const FEDRawData& l1tRcd = feds->FEDData(1024);

  const unsigned char* data = l1tRcd.data();
  FEDHeader header(data);

  bool mp = false;
  for (auto itr : mpList_) {
    mp |= (((header.bxID() - 1) % 9) == itr);
  }

  return mp;
}

//define this as a plug-in
DEFINE_FWK_MODULE(TMTFilter);
