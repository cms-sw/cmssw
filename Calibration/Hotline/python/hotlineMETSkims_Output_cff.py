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

OutALCARECOMETHotline = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring(
            "pathHotlineSkimPFMET",
            "pathHotlineSkimCaloMET",
            "pathHotlineSkimCondMET",
        ),
    ),
    outputCommands = cms.untracked.vstring(
        'drop *'
        )
)

#keep RAW event content
OutALCARECOMETHotline.outputCommands.extend(cms.untracked.vstring(
        'keep FEDRawDataCollection_rawDataCollector_*_*',
            'keep FEDRawDataCollection_source_*_*'
            ))

#keep RECO event content
OutALCARECOMETHotline.outputCommands.extend(RecoLocalTrackerRECO.outputCommands)
OutALCARECOMETHotline.outputCommands.extend(RecoLocalMuonRECO.outputCommands)
OutALCARECOMETHotline.outputCommands.extend(RecoLocalCaloRECO.outputCommands)
OutALCARECOMETHotline.outputCommands.extend(RecoEcalRECO.outputCommands)
OutALCARECOMETHotline.outputCommands.extend(TrackingToolsRECO.outputCommands)
OutALCARECOMETHotline.outputCommands.extend(RecoTrackerRECO.outputCommands)
OutALCARECOMETHotline.outputCommands.extend(RecoJetsRECO.outputCommands)
OutALCARECOMETHotline.outputCommands.extend(RecoMETRECO.outputCommands)
OutALCARECOMETHotline.outputCommands.extend(RecoMuonRECO.outputCommands)
OutALCARECOMETHotline.outputCommands.extend(RecoBTauRECO.outputCommands)
OutALCARECOMETHotline.outputCommands.extend(RecoBTagRECO.outputCommands)
OutALCARECOMETHotline.outputCommands.extend(RecoTauTagRECO.outputCommands)
OutALCARECOMETHotline.outputCommands.extend(RecoVertexRECO.outputCommands)
OutALCARECOMETHotline.outputCommands.extend(RecoEgammaRECO.outputCommands)
OutALCARECOMETHotline.outputCommands.extend(RecoPixelVertexingRECO.outputCommands)
OutALCARECOMETHotline.outputCommands.extend(RecoParticleFlowRECO.outputCommands)
OutALCARECOMETHotline.outputCommands.extend(BeamSpotRECO.outputCommands)
OutALCARECOMETHotline.outputCommands.extend(L1TriggerRECO.outputCommands)
OutALCARECOMETHotline.outputCommands.extend(HLTriggerRECO.outputCommands)
OutALCARECOMETHotline.outputCommands.extend(MEtoEDMConverterRECO.outputCommands)
OutALCARECOMETHotline.outputCommands.extend(EvtScalersRECO.outputCommands)
OutALCARECOMETHotline.outputCommands.extend(cms.untracked.vstring('keep *_logErrorHarvester_*_*'))
OutALCARECOMETHotline.outputCommands.extend(EITopPAGEventContent.outputCommands)
