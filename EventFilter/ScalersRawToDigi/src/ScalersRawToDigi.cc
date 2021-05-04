// -*- C++ -*-
//
// Package:    EventFilter/ScalersRawToDigi
// Class:      ScalersRawToDigi
//
/**\class ScalersRawToDigi ScalersRawToDigi.cc EventFilter/ScalersRawToDigi/src/ScalersRawToDigi.cc

 Description: Unpack FED data to Trigger and Lumi Scalers "bank"
 These Scalers are in FED id ScalersRaw::SCALERS_FED_ID

*/
//
// Original Author:  William Badgett
//         Created:  Wed Nov 14 07:47:59 CDT 2006
//

#include <memory>

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

// FEDRawData
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

// Scalers classes
#include "DataFormats/Scalers/interface/L1AcceptBunchCrossing.h"
#include "DataFormats/Scalers/interface/L1TriggerScalers.h"
#include "DataFormats/Scalers/interface/Level1TriggerScalers.h"
#include "DataFormats/Scalers/interface/Level1TriggerRates.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"
#include "DataFormats/Scalers/interface/BeamSpotOnline.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "DataFormats/Scalers/interface/ScalersRaw.h"

class ScalersRawToDigi : public edm::global::EDProducer<> {
public:
  explicit ScalersRawToDigi(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  const edm::EDGetTokenT<FEDRawDataCollection> fedToken_;
  const edm::EDPutTokenT<L1AcceptBunchCrossingCollection> bunchPutToken_;
  const edm::EDPutTokenT<L1TriggerScalersCollection> l1ScalerPutToken_;
  const edm::EDPutTokenT<Level1TriggerScalersCollection> lvl1ScalerPutToken_;
  const edm::EDPutTokenT<LumiScalersCollection> lumiScalerPutToken_;
  const edm::EDPutTokenT<BeamSpotOnlineCollection> beamSpotPutToken_;
  const edm::EDPutTokenT<DcsStatusCollection> dcsPutToken_;
};

// Constructor
ScalersRawToDigi::ScalersRawToDigi(const edm::ParameterSet& iConfig)
    : fedToken_{consumes<FEDRawDataCollection>(iConfig.getParameter<edm::InputTag>("scalersInputTag"))},
      bunchPutToken_{produces<L1AcceptBunchCrossingCollection>()},
      l1ScalerPutToken_{produces<L1TriggerScalersCollection>()},
      lvl1ScalerPutToken_{produces<Level1TriggerScalersCollection>()},
      lumiScalerPutToken_{produces<LumiScalersCollection>()},
      beamSpotPutToken_{produces<BeamSpotOnlineCollection>()},
      dcsPutToken_{produces<DcsStatusCollection>()} {}

void ScalersRawToDigi::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("scalersInputTag", edm::InputTag("rawDataCollector"));
  descriptions.add("scalersRawToDigi", desc);
}

// Method called to produce the data
void ScalersRawToDigi::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;

  // Get the FED data collection
  auto const& rawdata = iEvent.get(fedToken_);

  LumiScalersCollection pLumi;

  L1TriggerScalersCollection pOldTrigger;

  Level1TriggerScalersCollection pTrigger;

  L1AcceptBunchCrossingCollection pBunch;

  BeamSpotOnlineCollection pBeamSpotOnline;
  DcsStatusCollection pDcsStatus;

  /// Take a reference to this FED's data
  const FEDRawData& fedData = rawdata.FEDData(ScalersRaw::SCALERS_FED_ID);
  unsigned short int length = fedData.size();
  if (length > 0) {
    int nWords = length / 8;
    int nBytesExtra = 0;

    const ScalersEventRecordRaw_v6* raw = (struct ScalersEventRecordRaw_v6*)fedData.data();
    if ((raw->version == 1) || (raw->version == 2)) {
      pOldTrigger.emplace_back(fedData.data());
      nBytesExtra = length - sizeof(struct ScalersEventRecordRaw_v1);
    } else if (raw->version >= 3) {
      pTrigger.emplace_back(fedData.data());
      if (raw->version >= 6) {
        nBytesExtra = ScalersRaw::N_BX_v6 * sizeof(unsigned long long);
      } else {
        nBytesExtra = ScalersRaw::N_BX_v2 * sizeof(unsigned long long);
      }
    }

    pLumi.emplace_back(fedData.data());

    if ((nBytesExtra >= 8) && ((nBytesExtra % 8) == 0)) {
      unsigned long long const* data = (unsigned long long const*)fedData.data();

      int nWordsExtra = nBytesExtra / 8;
      for (int i = 0; i < nWordsExtra; i++) {
        int index = nWords - (nWordsExtra + 1) + i;
        pBunch.emplace_back(i, data[index]);
      }
    }

    if (raw->version >= 4) {
      pBeamSpotOnline.emplace_back(fedData.data());

      pDcsStatus.emplace_back(fedData.data());
    }
  }
  iEvent.emplace(l1ScalerPutToken_, std::move(pOldTrigger));
  iEvent.emplace(lvl1ScalerPutToken_, std::move(pTrigger));
  iEvent.emplace(lumiScalerPutToken_, std::move(pLumi));
  iEvent.emplace(bunchPutToken_, std::move(pBunch));
  iEvent.emplace(beamSpotPutToken_, std::move(pBeamSpotOnline));
  iEvent.emplace(dcsPutToken_, std::move(pDcsStatus));
}

// Define this as a plug-in
DEFINE_FWK_MODULE(ScalersRawToDigi);
