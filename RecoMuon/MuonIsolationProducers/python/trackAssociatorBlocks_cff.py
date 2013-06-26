import FWCore.ParameterSet.Config as cms

from TrackingTools.TrackAssociator.default_cfi import *


# -*-TCL-*-
MIsoTrackAssociatorDefault = cms.PSet(
    TrackAssociatorParameterBlock
)
MIsoTrackAssociatorTowers = cms.PSet(
    TrackAssociatorParameterBlock
)

MIsoTrackAssociatorHits = cms.PSet(
    TrackAssociatorParameterBlock
)

MIsoTrackAssociatorJets = cms.PSet(
    TrackAssociatorParameterBlock
)

MIsoTrackAssociatorDefault.TrackAssociatorParameters.useEcal = False ## RecoHits
MIsoTrackAssociatorDefault.TrackAssociatorParameters.useHcal = False ## RecoHits
MIsoTrackAssociatorDefault.TrackAssociatorParameters.useHO = False ## RecoHits
MIsoTrackAssociatorDefault.TrackAssociatorParameters.useCalo = True ## CaloTowers
MIsoTrackAssociatorDefault.TrackAssociatorParameters.useMuon = False ## RecoHits
MIsoTrackAssociatorDefault.TrackAssociatorParameters.usePreshower = False
MIsoTrackAssociatorDefault.TrackAssociatorParameters.dREcalPreselection = 1.0
MIsoTrackAssociatorDefault.TrackAssociatorParameters.dRHcalPreselection = 1.0
MIsoTrackAssociatorDefault.TrackAssociatorParameters.dREcal = 1.0
MIsoTrackAssociatorDefault.TrackAssociatorParameters.dRHcal = 1.0

MIsoTrackAssociatorTowers.TrackAssociatorParameters.useEcal = False ## RecoHits
MIsoTrackAssociatorTowers.TrackAssociatorParameters.useHcal = False ## RecoHits
MIsoTrackAssociatorTowers.TrackAssociatorParameters.useHO = False ## RecoHits
MIsoTrackAssociatorTowers.TrackAssociatorParameters.useCalo = True ## CaloTowers
MIsoTrackAssociatorTowers.TrackAssociatorParameters.useMuon = False ## RecoHits
MIsoTrackAssociatorTowers.TrackAssociatorParameters.usePreshower = False
MIsoTrackAssociatorTowers.TrackAssociatorParameters.dREcalPreselection = 1.0
MIsoTrackAssociatorTowers.TrackAssociatorParameters.dRHcalPreselection = 1.0
MIsoTrackAssociatorTowers.TrackAssociatorParameters.dREcal = 1.0
MIsoTrackAssociatorTowers.TrackAssociatorParameters.dRHcal = 1.0

MIsoTrackAssociatorHits.TrackAssociatorParameters.useEcal = True ## RecoHits
MIsoTrackAssociatorHits.TrackAssociatorParameters.useHcal = True ## RecoHits
MIsoTrackAssociatorHits.TrackAssociatorParameters.useHO = True ## RecoHits
MIsoTrackAssociatorHits.TrackAssociatorParameters.useCalo = False ## CaloTowers
MIsoTrackAssociatorHits.TrackAssociatorParameters.useMuon = False ## RecoHits
MIsoTrackAssociatorHits.TrackAssociatorParameters.usePreshower = False
MIsoTrackAssociatorHits.TrackAssociatorParameters.dREcalPreselection = 1.0
MIsoTrackAssociatorHits.TrackAssociatorParameters.dRHcalPreselection = 1.0
MIsoTrackAssociatorHits.TrackAssociatorParameters.dREcal = 1.0
MIsoTrackAssociatorHits.TrackAssociatorParameters.dRHcal = 1.0

MIsoTrackAssociatorJets.TrackAssociatorParameters.useEcal = False ## RecoHits
MIsoTrackAssociatorJets.TrackAssociatorParameters.useHcal = False ## RecoHits
MIsoTrackAssociatorJets.TrackAssociatorParameters.useHO = False ## RecoHits
MIsoTrackAssociatorJets.TrackAssociatorParameters.useCalo = True ## CaloTowers
MIsoTrackAssociatorJets.TrackAssociatorParameters.useMuon = False ## RecoHits
MIsoTrackAssociatorJets.TrackAssociatorParameters.usePreshower = False
MIsoTrackAssociatorJets.TrackAssociatorParameters.dREcalPreselection = 0.5
MIsoTrackAssociatorJets.TrackAssociatorParameters.dRHcalPreselection = 0.5
MIsoTrackAssociatorJets.TrackAssociatorParameters.dREcal = 0.5
MIsoTrackAssociatorJets.TrackAssociatorParameters.dRHcal = 0.5


