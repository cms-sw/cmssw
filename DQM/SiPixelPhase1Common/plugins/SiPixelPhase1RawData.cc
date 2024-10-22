// -*- C++ -*-
//
// Package:     SiPixelPhase1RawData
// Class:       SiPixelPhase1RawData
//

// Original Author: Marcel Schneider

#include "DQM/SiPixelPhase1Common/interface/SiPixelPhase1Base.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelRawDataError.h"
#include "FWCore/Framework/interface/MakerMacros.h"

namespace {

  class SiPixelPhase1RawData final : public SiPixelPhase1Base {
    enum { NERRORS, FIFOFULL, TBMMESSAGE, TBMTYPE, TYPE_NERRORS };

  public:
    explicit SiPixelPhase1RawData(const edm::ParameterSet& conf);
    void analyze(const edm::Event&, const edm::EventSetup&) override;

  private:
    edm::EDGetTokenT<edm::DetSetVector<SiPixelRawDataError>> srcToken_;
  };

  SiPixelPhase1RawData::SiPixelPhase1RawData(const edm::ParameterSet& iConfig) : SiPixelPhase1Base(iConfig) {
    srcToken_ = consumes<edm::DetSetVector<SiPixelRawDataError>>(iConfig.getParameter<edm::InputTag>("src"));
  }

  void SiPixelPhase1RawData::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    if (!checktrigger(iEvent, iSetup, DCS))
      return;

    edm::Handle<edm::DetSetVector<SiPixelRawDataError>> input;
    iEvent.getByToken(srcToken_, input);
    if (!input.isValid())
      return;

    for (auto it = input->begin(); it != input->end(); ++it) {
      for (auto& siPixelRawDataError : *it) {
        int fed = siPixelRawDataError.getFedId();
        int type = siPixelRawDataError.getType();
        DetId id = it->detId();

        // encoding of the channel number within the FED error word
        const uint32_t LINK_bits = 6;
        const uint32_t LINK_shift = 26;
        const uint64_t LINK_mask = (1 << LINK_bits) - 1;

        uint64_t errorWord = 0;
        // use 64bit word for some error types
        // invalid header, invalid trailer, size mismatch
        if (type == 32 || type == 33 || type == 34) {
          errorWord = siPixelRawDataError.getWord64();
        } else {
          errorWord = siPixelRawDataError.getWord32();
        }

        int32_t chanNmbr = (errorWord >> LINK_shift) & LINK_mask;
        // timeout
        if (type == 29)
          chanNmbr = -1;  // TODO: different formula needed.

        uint32_t error_data = errorWord & 0xFF;

        if (type == 28) {  // overflow.
          for (uint32_t i = 0; i < 8; i++) {
            if (error_data & (1 << i))
              histo[FIFOFULL].fill(i, id, &iEvent, fed, chanNmbr);
          }
        }

        if (type == 30) {                                      // TBM stuff.
          uint32_t statemachine_state = errorWord >> 8 & 0xF;  // next 4 bits after data
          const uint32_t tbm_types[16] = {0, 1, 2, 4, 2, 4, 2, 4, 3, 1, 4, 4, 4, 4, 4, 4};

          histo[TBMTYPE].fill(tbm_types[statemachine_state], id, &iEvent, fed, chanNmbr);

          for (uint32_t i = 0; i < 8; i++) {
            if (error_data & (1 << i))
              histo[TBMMESSAGE].fill(i, id, &iEvent, fed, chanNmbr);
          }
          continue;  // we don't really consider these as errors.
        }

        // note that a DetId of 0xFFFFFFFF can mean 'no DetId'.
        // We hijack column and row for FED and chan in this case,
        // the GeometryInterface does understand that.

        histo[NERRORS].fill(id, &iEvent, fed, chanNmbr);
        histo[TYPE_NERRORS].fill(type, id, &iEvent, fed, chanNmbr);
      }
    }

    histo[NERRORS].executePerEventHarvesting(&iEvent);
  }

}  //namespace

DEFINE_FWK_MODULE(SiPixelPhase1RawData);
