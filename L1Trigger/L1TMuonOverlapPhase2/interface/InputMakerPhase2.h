/*
 * InputMakerPhase2.h
 *
 *  Created on: May 20, 2020
 *      Author: kbunkow
 */

#ifndef L1Trigger_L1TMuonOverlapPhase2_InputMakerPhase2_h
#define L1Trigger_L1TMuonOverlapPhase2_InputMakerPhase2_h

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTPhContainer.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFConfiguration.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/MuonStub.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/MuonStubMakerBase.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFinputMaker.h"
#include "L1Trigger/L1TMuonOverlapPhase2/interface/OmtfPhase2AngleConverter.h"

class DtPhase2DigiToStubsConverter : public DigiToStubsConverterBase {
public:
  DtPhase2DigiToStubsConverter(edm::EDGetTokenT<L1Phase2MuDTPhContainer> inputTokenDtPh,
                               edm::EDGetTokenT<L1MuDTChambThContainer> inputTokenDtTh)
      : inputTokenDtPh(inputTokenDtPh), inputTokenDtTh(inputTokenDtTh){};

  ~DtPhase2DigiToStubsConverter() override{};

  void loadDigis(const edm::Event& event) override;

  void makeStubs(MuonStubPtrs2D& muonStubsInLayers,
                 unsigned int iProcessor,
                 l1t::tftype procTyp,
                 int bxFrom,
                 int bxTo,
                 std::vector<std::unique_ptr<IOMTFEmulationObserver> >& observers) override;

  //dtThDigis is provided as argument, because in the OMTF implementation the phi and eta digis are merged (even thought it is artificial)
  virtual void addDTphiDigi(MuonStubPtrs2D& muonStubsInLayers,
                            const L1Phase2MuDTPhDigi& digi,
                            const L1MuDTChambThContainer* dtThDigis,
                            unsigned int iProcessor,
                            l1t::tftype procTyp) = 0;

  virtual void addDTetaStubs(MuonStubPtrs2D& muonStubsInLayers,
                             const L1MuDTChambThDigi& thetaDigi,
                             unsigned int iProcessor,
                             l1t::tftype procTyp) = 0;

  virtual bool acceptDigi(const DTChamberId& dTChamberId, unsigned int iProcessor, l1t::tftype procType) {
    return true;
  }

protected:
  bool mergePhiAndTheta = true;

  edm::EDGetTokenT<L1Phase2MuDTPhContainer> inputTokenDtPh;
  edm::EDGetTokenT<L1MuDTChambThContainer> inputTokenDtTh;

  edm::Handle<L1Phase2MuDTPhContainer> dtPhDigis;
  edm::Handle<L1MuDTChambThContainer> dtThDigis;
};

class DtPhase2DigiToStubsConverterOmtf : public DtPhase2DigiToStubsConverter {
public:
  DtPhase2DigiToStubsConverterOmtf(const OMTFConfiguration* config,
                                   const OmtfAngleConverter* angleConverter,
                                   edm::EDGetTokenT<L1Phase2MuDTPhContainer> inputTokenDtPh,
                                   edm::EDGetTokenT<L1MuDTChambThContainer> inputTokenDtTh)
      : DtPhase2DigiToStubsConverter(inputTokenDtPh, inputTokenDtTh),
        config(*config),
        angleConverter(*angleConverter){};

  ~DtPhase2DigiToStubsConverterOmtf() override = default;

  //dtThDigis is provided as argument, because in the OMTF implementation the phi and eta digis are merged (even thought it is artificial)
  void addDTphiDigi(MuonStubPtrs2D& muonStubsInLayers,
                    const L1Phase2MuDTPhDigi& digi,
                    const L1MuDTChambThContainer* dtThDigis,
                    unsigned int iProcessor,
                    l1t::tftype procTyp) override;

  void addDTetaStubs(MuonStubPtrs2D& muonStubsInLayers,
                     const L1MuDTChambThDigi& thetaDigi,
                     unsigned int iProcessor,
                     l1t::tftype procTyp) override;

  bool acceptDigi(const DTChamberId& dTChamberId, unsigned int iProcessor, l1t::tftype procType) override;

private:
  const OMTFConfiguration& config;
  const OmtfAngleConverter& angleConverter;
};

class InputMakerPhase2 : public OMTFinputMaker {
public:
  InputMakerPhase2(const edm::ParameterSet& edmParameterSet,
                   MuStubsInputTokens& muStubsInputTokens,
                   edm::EDGetTokenT<L1Phase2MuDTPhContainer> inputTokenDTPhPhase2,
                   const OMTFConfiguration* config,
                   std::unique_ptr<OmtfAngleConverter> angleConverter);

  ~InputMakerPhase2() override = default;

  //the phi and eta digis are merged (even thought it is artificial)
  virtual void addDTphiDigi(MuonStubPtrs2D& muonStubsInLayers,
                            const L1Phase2MuDTPhDigi& digi,
                            const L1Phase2MuDTPhContainer* dtThDigis,
                            unsigned int iProcessor,
                            l1t::tftype procTyp) {}
};

#endif /* L1Trigger_L1TMuonOverlapPhase2_InputMakerPhase2_h */
