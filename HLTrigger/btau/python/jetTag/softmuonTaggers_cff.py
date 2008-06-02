import FWCore.ParameterSet.Config as cms

from HLTrigger.Muon.CommonModules_cff import *
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
from RecoBTau.JetTagComputer.jetTagRecord_cfi import *
from RecoBTag.SoftLepton.softLeptonByDistanceES_cfi import *
from RecoBTag.SoftLepton.softLeptonByPtES_cfi import *
hltBSoftmuonL25TagInfos = cms.EDFilter("SoftLepton",
    refineJetAxis = cms.uint32(0),
    primaryVertex = cms.InputTag("nominal"),
    leptons = cms.InputTag("hltL2Muons"),
    leptonQualityCut = cms.double(0.0),
    jets = cms.InputTag("hltBSoftmuonL25Jets"),
    leptonDeltaRCut = cms.double(0.4),
    leptonChi2Cut = cms.double(0.0)
)

hltBSoftmuonL25BJetTags = cms.EDProducer("JetTagProducer",
    tagInfo = cms.InputTag("hltBSoftmuonL25TagInfos"),
    jetTagComputer = cms.string('softLeptonByDistance')
)

hltBSoftmuonL3TagInfos = cms.EDFilter("SoftLepton",
    refineJetAxis = cms.uint32(0),
    primaryVertex = cms.InputTag("nominal"),
    leptons = cms.InputTag("hltL3Muons"),
    leptonQualityCut = cms.double(0.0),
    jets = cms.InputTag("hltBSoftmuonL25Jets"),
    leptonDeltaRCut = cms.double(0.4),
    leptonChi2Cut = cms.double(0.0)
)

hltBSoftmuonL3BJetTags = cms.EDProducer("JetTagProducer",
    tagInfo = cms.InputTag("hltBSoftmuonL3TagInfos"),
    jetTagComputer = cms.string('softLeptonByPt')
)

hltBSoftmuonL3BJetTagsByDR = cms.EDProducer("JetTagProducer",
    tagInfo = cms.InputTag("hltBSoftmuonL3TagInfos"),
    jetTagComputer = cms.string('softLeptonByDistance')
)


