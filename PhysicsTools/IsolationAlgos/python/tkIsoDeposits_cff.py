import FWCore.ParameterSet.Config as cms

from Geometry.CMSCommonData.cmsIdealGeometryXML_cff import *
from MagneticField.Engine.volumeBasedMagneticField_cfi import *
from Geometry.CaloEventSetup.CaloGeometry_cfi import *
from Geometry.CommonDetUnit.globalTrackingGeometry_cfi import *
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi import *
from TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff import *
from RecoMuon.MuonIsolationProducers.trackExtractorBlocks_cff import *
from RecoMuon.MuonIsolationProducers.caloExtractorByAssociatorBlocks_cff import *
from RecoMuon.MuonIsolationProducers.jetExtractorBlock_cff import *
from RecoMuon.MuonIsolationProducers.caloExtractorBlocks_cff import *
from PhysicsTools.RecoAlgos.highPtTracks_cfi import *

import PhysicsTools.IsolationAlgos.test.mu.candIsoDepositCtfTk_cfi
tkIsoDepositTk = PhysicsTools.IsolationAlgos.test.mu.candIsoDepositCtfTk_cfi.candIsoDepositCtfTk.clone()

tkIsoDepositTk.src = 'highPtTracks'
tkIsoDepositTk.trackType = 'best'
tkIsoDepositTk.ExtractorPSet = cms.PSet(
    MIsoTrackExtractorBlock
)

import PhysicsTools.IsolationAlgos.test.mu.candIsoDepositCalByAssociatorTowers_cfi
tkIsoDepositCalByAssociatorTowers = PhysicsTools.IsolationAlgos.test.mu.candIsoDepositCalByAssociatorTowers_cfi.candIsoDepositCalByAssociatorTowers.clone()

tkIsoDepositCalByAssociatorTowers.src = 'highPtTracks'
tkIsoDepositCalByAssociatorTowers.trackType = 'best'

import PhysicsTools.IsolationAlgos.test.mu.candIsoDepositCalByAssociatorHits_cfi
tkIsoDepositCalByAssociatorHits = PhysicsTools.IsolationAlgos.test.mu.candIsoDepositCalByAssociatorHits_cfi.candIsoDepositCalByAssociatorHits.clone()

tkIsoDepositCalByAssociatorHits.src = 'highPtTracks'
tkIsoDepositCalByAssociatorHits.trackType = 'best'

import PhysicsTools.IsolationAlgos.test.mu.candIsoDepositJets_cfi
tkIsoDepositJets = PhysicsTools.IsolationAlgos.test.mu.candIsoDepositJets_cfi.candIsoDepositJets.clone()

tkIsoDepositJets.src = 'highPtTracks'
tkIsoDepositJets.trackType = 'best'

import PhysicsTools.IsolationAlgos.test.mu.candIsoDepositCal_cfi
tkIsoDepositCalEcal = PhysicsTools.IsolationAlgos.test.mu.candIsoDepositCal_cfi.candIsoDepositCal.clone()

tkIsoDepositCalEcal.src = 'highPtTracks'
tkIsoDepositCalEcal.trackType = 'best'
tkIsoDepositCalEcal.ExtractorPSet = cms.PSet(
    MIsoCaloExtractorEcalBlock
)

import PhysicsTools.IsolationAlgos.test.mu.candIsoDepositCal_cfi
tkIsoDepositCalHcal = PhysicsTools.IsolationAlgos.test.mu.candIsoDepositCal_cfi.candIsoDepositCal.clone()

tkIsoDepositCalHcal.src = 'highPtTracks'
tkIsoDepositCalHcal.trackType = 'best'
tkIsoDepositCalHcal.ExtractorPSet = cms.PSet(
    MIsoCaloExtractorHcalBlock
)

tkIsoDeposits = cms.Sequence(highPtTracks+tkIsoDepositTk+tkIsoDepositCalByAssociatorTowers+tkIsoDepositJets)



