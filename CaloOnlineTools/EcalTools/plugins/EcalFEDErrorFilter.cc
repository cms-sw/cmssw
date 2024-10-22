// -*- C++ -*-
//
// Package:    EcalFEDErrorFilter
// Class:      EcalFEDErrorFilter
//
/**\class EcalFEDErrorFilter EcalFEDErrorFilter.cc filter/EcalFEDErrorFilter/src/EcalFEDErrorFilter.cc

Description: <one line class summary>

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Giovanni FRANZONI
//         Created:  Tue Jan 22 13:55:00 CET 2008
//
//

#include "CaloOnlineTools/EcalTools/plugins/EcalFEDErrorFilter.h"

//
// constructors and destructor
//
EcalFEDErrorFilter::EcalFEDErrorFilter(const edm::ParameterSet& iConfig)
    : HLTFilter(iConfig),
      dataToken_(consumes<FEDRawDataCollection>(iConfig.getParameter<edm::InputTag>("InputLabel"))) {
  //now do what ever initialization is needed
  fedUnpackList_ = iConfig.getUntrackedParameter<std::vector<int> >("FEDs", std::vector<int>());
  if (fedUnpackList_.empty())
    for (int i = FEDNumbering::MINECALFEDID; i <= FEDNumbering::MAXECALFEDID; i++)
      fedUnpackList_.push_back(i);
}

EcalFEDErrorFilter::~EcalFEDErrorFilter() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool EcalFEDErrorFilter::hltFilter(edm::Event& iEvent,
                                   const edm::EventSetup& iSetup,
                                   trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  using namespace edm;

  const edm::Handle<FEDRawDataCollection>& rawdata = iEvent.getHandle(dataToken_);

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
      if (crcError) {
        edm::LogVerbatim("EcalTools") << "CRCERROR in FED " << *i << " trailer is " << std::setw(8) << std::hex
                                      << (*fedTrailer) << std::dec;
        return true;
      }
    }
  }

  return false;
}
