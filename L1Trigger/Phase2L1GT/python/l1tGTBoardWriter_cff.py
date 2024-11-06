import FWCore.ParameterSet.Config as cms
from L1Trigger.Phase2L1GT.l1tGTProducer_cff import l1tGTProducer

BoardDataInput = cms.EDAnalyzer("L1GTObjectBoardWriter",
    GTTPromptJets = cms.untracked.InputTag(l1tGTProducer.GTTPromptJets.value()),
    GTTDisplacedJets = cms.untracked.InputTag(l1tGTProducer.GTTDisplacedJets.value()),
    GTTPromptHtSum = cms.untracked.InputTag(l1tGTProducer.GTTPromptHtSum.value()),
    GTTDisplacedHtSum = cms.untracked.InputTag(l1tGTProducer.GTTDisplacedHtSum.value()),
    GTTEtSum = cms.untracked.InputTag(l1tGTProducer.GTTEtSum.value()),
    GTTPrimaryVert = cms.untracked.InputTag(l1tGTProducer.GTTPrimaryVert.value()),
    GMTSaPromptMuons = cms.untracked.InputTag(l1tGTProducer.GMTSaPromptMuons.value()),
    GMTSaDisplacedMuons = cms.untracked.InputTag(l1tGTProducer.GMTSaDisplacedMuons.value()),
    GMTTkMuons = cms.untracked.InputTag(l1tGTProducer.GMTTkMuons.value()),
    CL2JetsSC4 = cms.untracked.InputTag(l1tGTProducer.CL2JetsSC4.value()),
    CL2JetsSC8 = cms.untracked.InputTag(l1tGTProducer.CL2JetsSC8.value()),
    CL2Electrons = cms.untracked.InputTag(l1tGTProducer.CL2Electrons.value()),
    CL2Photons = cms.untracked.InputTag(l1tGTProducer.CL2Photons.value()),
    CL2Taus = cms.untracked.InputTag(l1tGTProducer.CL2Taus.value()),
    CL2EtSum = cms.untracked.InputTag(l1tGTProducer.CL2EtSum.value()),
    CL2HtSum = cms.untracked.InputTag(l1tGTProducer.CL2HtSum.value()),
    filename = cms.untracked.string("inputPattern"),
    bufferFileType = cms.untracked.string("input")
)

BoardDataOutputObjects = cms.EDAnalyzer("L1GTObjectBoardWriter",
    GTTPromptJets = cms.untracked.InputTag(l1tGTProducer.GTTPromptJets.value()),
    GTTDisplacedJets = cms.untracked.InputTag(l1tGTProducer.GTTDisplacedJets.value()),
    GTTPromptHtSum = cms.untracked.InputTag(l1tGTProducer.GTTPromptHtSum.value()),
    GTTDisplacedHtSum = cms.untracked.InputTag(l1tGTProducer.GTTDisplacedHtSum.value()),
    GTTEtSum = cms.untracked.InputTag(l1tGTProducer.GTTEtSum.value()),
    GTTPrimaryVert = cms.untracked.InputTag(l1tGTProducer.GTTPrimaryVert.value()),
    GMTSaPromptMuons = cms.untracked.InputTag(l1tGTProducer.GMTSaPromptMuons.value()),
    GMTSaDisplacedMuons = cms.untracked.InputTag(l1tGTProducer.GMTSaDisplacedMuons.value()),
    GMTTkMuons = cms.untracked.InputTag(l1tGTProducer.GMTTkMuons.value()),
    CL2JetsSC4 = cms.untracked.InputTag(l1tGTProducer.CL2JetsSC4.value()),
    CL2JetsSC8 = cms.untracked.InputTag(l1tGTProducer.CL2JetsSC8.value()),
    CL2Electrons = cms.untracked.InputTag(l1tGTProducer.CL2Electrons.value()),
    CL2Photons = cms.untracked.InputTag(l1tGTProducer.CL2Photons.value()),
    CL2Taus = cms.untracked.InputTag(l1tGTProducer.CL2Taus.value()),
    CL2EtSum = cms.untracked.InputTag(l1tGTProducer.CL2EtSum.value()),
    CL2HtSum = cms.untracked.InputTag(l1tGTProducer.CL2HtSum.value()),
    filename = cms.untracked.string("outputObjectPattern"),
    bufferFileType = cms.untracked.string("output")
)

AlgoBitBoardData = cms.EDAnalyzer("L1GTAlgoBoardWriter",
    filename = cms.untracked.string("algoBitPattern"),
    algoBlocksTag = cms.untracked.InputTag("l1tGTAlgoBlockProducer"),
)
