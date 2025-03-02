#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/L1TrackTrigger/interface/TTDTC.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"

#include <string>

using namespace std;
using namespace edm;
using namespace tt;

namespace trackerTFP {

  /*! \class  trackerTFP::ProducerPP
   *  \brief  L1TrackTrigger PatchPanel between DTC and TFP emulator
   *  \author Thomas Schuh
   *  \date   2023, April
   */
  class ProducerPP : public stream::EDProducer<> {
  public:
    explicit ProducerPP(const ParameterSet&);
    ~ProducerPP() override {}

  private:
    void produce(Event&, const EventSetup&) override;
    // ED input token of DTC stubs
    EDGetTokenT<TTDTC> edGetToken_;
    // ED output token for accepted stubs
    EDPutTokenT<StreamsStub> edPutToken_;
    // Setup token
    ESGetToken<Setup, SetupRcd> esGetTokenSetup_;
  };

  ProducerPP::ProducerPP(const ParameterSet& iConfig) {
    const string& label = iConfig.getParameter<string>("InputLabelPP");
    const string& branch = iConfig.getParameter<string>("BranchStubs");
    // book in- and output ED products
    edGetToken_ = consumes<TTDTC>(InputTag(label, branch));
    edPutToken_ = produces<StreamsStub>(branch);
    // book ES products
    esGetTokenSetup_ = esConsumes();
  }

  void ProducerPP::produce(Event& iEvent, const EventSetup& iSetup) {
    // helper classe to store configurations
    const Setup* setup = &iSetup.getData(esGetTokenSetup_);
    // empty GP products
    StreamsStub stubs(setup->numRegions() * setup->numDTCsPerTFP());
    // read in DTC Product and produce TFP product
    const TTDTC& ttDTC = iEvent.get(edGetToken_);
    for (int region = 0; region < setup->numRegions(); region++) {
      const int offset = region * setup->numDTCsPerTFP();
      for (int channel = 0; channel < setup->numDTCsPerTFP(); channel++)
        stubs[offset + channel] = ttDTC.stream(region, channel);
    }
    // store products
    iEvent.emplace(edPutToken_, std::move(stubs));
  }

}  // namespace trackerTFP

DEFINE_FWK_MODULE(trackerTFP::ProducerPP);