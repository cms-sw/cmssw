#ifndef L1T_OmtfP1_MUONSTUBMAKERBASE_H
#define L1T_OmtfP1_MUONSTUBMAKERBASE_H

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiCollection.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/MuonStub.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/RpcClusterization.h"
#include <cstdint>
#include <memory>
#include <vector>

class ProcConfigurationBase;

namespace edm {
  class EventSetup;
}

struct MuStubsInputTokens {
  edm::EDGetTokenT<L1MuDTChambPhContainer> inputTokenDtPh;
  edm::EDGetTokenT<L1MuDTChambThContainer> inputTokenDtTh;
  edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection> inputTokenCSC;
  edm::EDGetTokenT<RPCDigiCollection> inputTokenRPC;
};

class DigiToStubsConverterBase {
public:
  virtual ~DigiToStubsConverterBase(){};

  //virtual void initialize(const edm::ParameterSet& edmCfg, const edm::EventSetup& es, const ProcConfigurationBase* procConf) = 0;

  virtual void loadDigis(const edm::Event& event) = 0;

  virtual void makeStubs(
      MuonStubPtrs2D& muonStubsInLayers, unsigned int iProcessor, l1t::tftype procTyp, int bxFrom, int bxTo) = 0;
};

class DtDigiToStubsConverter : public DigiToStubsConverterBase {
public:
  DtDigiToStubsConverter(edm::EDGetTokenT<L1MuDTChambPhContainer> inputTokenDtPh,
                         edm::EDGetTokenT<L1MuDTChambThContainer> inputTokenDtTh)
      : inputTokenDtPh(inputTokenDtPh), inputTokenDtTh(inputTokenDtTh){};

  ~DtDigiToStubsConverter() override{};

  //virtual void initialize(const edm::ParameterSet& edmCfg, const edm::EventSetup& es, const ProcConfigurationBase* procConf) {} //TODO is it needed at all?

  void loadDigis(const edm::Event& event) override;

  void makeStubs(
      MuonStubPtrs2D& muonStubsInLayers, unsigned int iProcessor, l1t::tftype procTyp, int bxFrom, int bxTo) override;

  //dtThDigis is provided as argument, because in the OMTF implementation the phi and eta digis are merged (even thought it is artificial)
  virtual void addDTphiDigi(MuonStubPtrs2D& muonStubsInLayers,
                            const L1MuDTChambPhDigi& digi,
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

  edm::EDGetTokenT<L1MuDTChambPhContainer> inputTokenDtPh;
  edm::EDGetTokenT<L1MuDTChambThContainer> inputTokenDtTh;

  edm::Handle<L1MuDTChambPhContainer> dtPhDigis;
  edm::Handle<L1MuDTChambThContainer> dtThDigis;
};

class CscDigiToStubsConverter : public DigiToStubsConverterBase {
public:
  CscDigiToStubsConverter(const ProcConfigurationBase* config,
                          edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection> inputTokenCsc)
      : config(config), inputTokenCsc(inputTokenCsc){};

  ~CscDigiToStubsConverter() override{};

  //virtual void initialize(const edm::ParameterSet& edmCfg, const edm::EventSetup& es, const ProcConfigurationBase* procConf) {} //TODO is it needed at all?

  void loadDigis(const edm::Event& event) override { event.getByToken(inputTokenCsc, cscDigis); }

  void makeStubs(
      MuonStubPtrs2D& muonStubsInLayers, unsigned int iProcessor, l1t::tftype procTyp, int bxFrom, int bxTo) override;

  //can add both phi and eta stubs
  virtual void addCSCstubs(MuonStubPtrs2D& muonStubsInLayers,
                           unsigned int rawid,
                           const CSCCorrelatedLCTDigi& digi,
                           unsigned int iProcessor,
                           l1t::tftype procTyp) = 0;

  virtual bool acceptDigi(const CSCDetId& cscDetId, unsigned int iProcessor, l1t::tftype procType) { return true; }

protected:
  const ProcConfigurationBase* config;

  bool mergePhiAndTheta = true;

  edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection> inputTokenCsc;
  edm::Handle<CSCCorrelatedLCTDigiCollection> cscDigis;
};

class RpcDigiToStubsConverter : public DigiToStubsConverterBase {
public:
  RpcDigiToStubsConverter(const ProcConfigurationBase* config,
                          edm::EDGetTokenT<RPCDigiCollection> inputTokenRpc,
                          const RpcClusterization* rpcClusterization)
      : config(config), inputTokenRpc(inputTokenRpc), rpcClusterization(rpcClusterization){};

  ~RpcDigiToStubsConverter() override{};

  //virtual void initialize(const edm::ParameterSet& edmCfg, const edm::EventSetup& es, const ProcConfigurationBase* procConf) {} //TODO is it needed at all?

  void loadDigis(const edm::Event& event) override { event.getByToken(inputTokenRpc, rpcDigis); }

  void makeStubs(
      MuonStubPtrs2D& muonStubsInLayers, unsigned int iProcessor, l1t::tftype procTyp, int bxFrom, int bxTo) override;

  virtual void addRPCstub(MuonStubPtrs2D& muonStubsInLayers,
                          const RPCDetId& roll,
                          const RpcCluster& cluster,
                          unsigned int iProcessor,
                          l1t::tftype procTyp) = 0;

  virtual bool acceptDigi(const RPCDetId& rpcDetId, unsigned int iProcessor, l1t::tftype procType) { return true; }

protected:
  const ProcConfigurationBase* config;

  bool mergePhiAndTheta = true;

  edm::EDGetTokenT<RPCDigiCollection> inputTokenRpc;
  edm::Handle<RPCDigiCollection> rpcDigis;

  const RpcClusterization* rpcClusterization;
};

//forward declaration - MuonGeometryTokens is defined and used in the AngleConverterBase
struct MuonGeometryTokens;

class MuonStubMakerBase {
public:
  MuonStubMakerBase(const ProcConfigurationBase* procConf);

  virtual ~MuonStubMakerBase();

  virtual void initialize(const edm::ParameterSet& edmCfg,
                          const edm::EventSetup& es,
                          const MuonGeometryTokens& muonGeometryTokens);

  void loadAndFilterDigis(const edm::Event& event);

  ///Method translating trigger digis into input matrix with global phi coordinates, fills the muonStubsInLayers
  void buildInputForProcessor(
      MuonStubPtrs2D& muonStubsInLayers, unsigned int iProcessor, l1t::tftype procTyp, int bxFrom = 0, int bxTo = 0);

protected:
  const ProcConfigurationBase* config = nullptr;

  std::vector<std::unique_ptr<DigiToStubsConverterBase> > digiToStubsConverters;

  RpcClusterization rpcClusterization;
};

#endif
