import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.Configuration.RecoLocalTracker_EventContent_cff import *
from RecoLocalMuon.Configuration.RecoLocalMuon_EventContent_cff import *
from RecoLocalCalo.Configuration.RecoLocalCalo_EventContent_cff import *
from RecoEcal.Configuration.RecoEcal_EventContent_cff import *
from TrackingTools.Configuration.TrackingTools_EventContent_cff import *
from RecoTracker.Configuration.RecoTracker_EventContent_cff import *
from RecoJets.Configuration.RecoJets_EventContent_cff import *
from RecoMET.Configuration.RecoMET_EventContent_cff import *
from RecoMuon.Configuration.RecoMuon_EventContent_cff import *
from RecoBTau.Configuration.RecoBTau_EventContent_cff import *
from RecoBTag.Configuration.RecoBTag_EventContent_cff import *
from RecoTauTag.Configuration.RecoTauTag_EventContent_cff import *
from RecoVertex.Configuration.RecoVertex_EventContent_cff import *
from RecoPixelVertexing.Configuration.RecoPixelVertexing_EventContent_cff import *
from RecoEgamma.Configuration.RecoEgamma_EventContent_cff import *
from RecoParticleFlow.Configuration.RecoParticleFlow_EventContent_cff import *
from L1Trigger.Configuration.L1Trigger_EventContent_cff import *
from RecoVertex.BeamSpotProducer.BeamSpot_EventContent_cff import *
from CommonTools.ParticleFlow.EITopPAG_EventContent_cff import EITopPAGEventContent
from EventFilter.ScalersRawToDigi.Scalers_EventContent_cff import *
from EventFilter.Configuration.DigiToRaw_EventContent_cff import *
from HLTrigger.Configuration.HLTrigger_EventContent_cff import *
from DQMOffline.Configuration.DQMOffline_EventContent_cff import *

OutALCARECOHotline = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring(
            "pathHotlineSkimSingleMuon",
            "pathHotlineSkimDoubleMuon",
            "pathHotlineSkimTripleMuon",
            "pathHotlineSkimSingleElectron",
            "pathHotlineSkimDoubleElectron",
            "pathHotlineSkimTripleElectron",
            "pathHotlineSkimSinglePhoton",
            "pathHotlineSkimDoublePhoton",
            "pathHotlineSkimTriplePhoton",
            "pathHotlineSkimSingleJet",
            "pathHotlineSkimDoubleJet",
            "pathHotlineSkimMultiJet",
            "pathHotlineSkimHT",
            "pathHotlineSkimMassiveDimuon",
            "pathHotlineSkimMassiveDielectron",
            "pathHotlineSkimMassiveEMu"
        ),
    ),
    outputCommands = cms.untracked.vstring(
        'drop *'
        )
)

#keep RAW event content
OutALCARECOHotline.outputCommands.extend(cms.untracked.vstring(
        'keep FEDRawDataCollection_rawDataCollector_*_*',
            'keep FEDRawDataCollection_source_*_*'
            ))

#keep RECO event content
OutALCARECOHotline.outputCommands.extend(RecoLocalTrackerRECO.outputCommands)
OutALCARECOHotline.outputCommands.extend(RecoLocalMuonRECO.outputCommands)
OutALCARECOHotline.outputCommands.extend(RecoLocalCaloRECO.outputCommands)
OutALCARECOHotline.outputCommands.extend(RecoEcalRECO.outputCommands)
OutALCARECOHotline.outputCommands.extend(TrackingToolsRECO.outputCommands)
OutALCARECOHotline.outputCommands.extend(RecoTrackerRECO.outputCommands)
OutALCARECOHotline.outputCommands.extend(RecoJetsRECO.outputCommands)
OutALCARECOHotline.outputCommands.extend(RecoMETRECO.outputCommands)
OutALCARECOHotline.outputCommands.extend(RecoMuonRECO.outputCommands)
OutALCARECOHotline.outputCommands.extend(RecoBTauRECO.outputCommands)
OutALCARECOHotline.outputCommands.extend(RecoBTagRECO.outputCommands)
OutALCARECOHotline.outputCommands.extend(RecoTauTagRECO.outputCommands)
OutALCARECOHotline.outputCommands.extend(RecoVertexRECO.outputCommands)
OutALCARECOHotline.outputCommands.extend(RecoEgammaRECO.outputCommands)
OutALCARECOHotline.outputCommands.extend(RecoPixelVertexingRECO.outputCommands)
OutALCARECOHotline.outputCommands.extend(RecoParticleFlowRECO.outputCommands)
OutALCARECOHotline.outputCommands.extend(BeamSpotRECO.outputCommands)
OutALCARECOHotline.outputCommands.extend(L1TriggerRECO.outputCommands)
OutALCARECOHotline.outputCommands.extend(HLTriggerRECO.outputCommands)
OutALCARECOHotline.outputCommands.extend(MEtoEDMConverterRECO.outputCommands)
OutALCARECOHotline.outputCommands.extend(EvtScalersRECO.outputCommands)
OutALCARECOHotline.outputCommands.extend(cms.untracked.vstring('keep *_logErrorHarvester_*_*'))
OutALCARECOHotline.outputCommands.extend(EITopPAGEventContent.outputCommands)
