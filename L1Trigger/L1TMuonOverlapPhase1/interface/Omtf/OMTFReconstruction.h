#ifndef L1T_OmtfP1_OMTFReconstruction_H
#define L1T_OmtfP1_OMTFReconstruction_H

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GhostBuster.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/IOMTFEmulationObserver.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/IProcessorEmulator.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFinputMaker.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFProcessor.h"

#include "CondFormats/DataRecord/interface/L1TMuonOverlapParamsRcd.h"

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

class OMTFConfiguration;
class OMTFConfigMaker;

class OMTFReconstruction {
public:
  OMTFReconstruction(const edm::ParameterSet&, MuStubsInputTokens& muStubsInputTokens);

  virtual ~OMTFReconstruction();

  void beginJob();

  void endJob();

  void beginRun(edm::Run const& run,
                edm::EventSetup const& iSetup,
                edm::ESGetToken<L1TMuonOverlapParams, L1TMuonOverlapParamsRcd>& omtfParamsEsToken,
                const MuonGeometryTokens& muonGeometryTokens,
                const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord>& magneticFieldEsToken,
                const edm::ESGetToken<Propagator, TrackingComponentsRecord>& propagatorEsToken);

  std::unique_ptr<l1t::RegionalMuonCandBxCollection> reconstruct(const edm::Event&, const edm::EventSetup&);

  //takes the ownership of the inputMaker
  void setInputMaker(unique_ptr<OMTFinputMaker> inputMaker) { this->inputMaker = std::move(inputMaker); }

  void virtual addObservers(const MuonGeometryTokens& muonGeometryTokens,
                            const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord>& magneticFieldEsToken,
                            const edm::ESGetToken<Propagator, TrackingComponentsRecord>& propagatorEsToken);

protected:
  edm::ParameterSet edmParameterSet;

  MuStubsInputTokens& muStubsInputTokens;

  int bxMin, bxMax;

  ///OMTF objects
  unique_ptr<OMTFConfiguration> omtfConfig;

  unique_ptr<OMTFinputMaker> inputMaker;

  unique_ptr<IProcessorEmulator> omtfProc;

  OMTFConfigMaker* m_OMTFConfigMaker;

  std::vector<std::unique_ptr<IOMTFEmulationObserver> > observers;

  edm::ESWatcher<L1TMuonOverlapParamsRcd> omtfParamsRecordWatcher;
};

#endif
