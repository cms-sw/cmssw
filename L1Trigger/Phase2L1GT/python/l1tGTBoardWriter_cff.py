import FWCore.ParameterSet.Config as cms
from L1Trigger.Phase2L1GT.l1tGTProducer_cff import l1tGTProducer

BoardDataInput = cms.EDAnalyzer("L1GTObjectBoardWriter",
    GTTPromptJets = l1tGTProducer.GTTPromptJets,
    GTTDisplacedJets = l1tGTProducer.GTTDisplacedJets,
    GTTPromptHtSum = l1tGTProducer.GTTPromptHtSum,
    GTTDisplacedHtSum = l1tGTProducer.GTTDisplacedHtSum,
    GTTEtSum = l1tGTProducer.GTTEtSum,
    GTTPrimaryVert = l1tGTProducer.GTTPrimaryVert,
    GMTSaPromptMuons = l1tGTProducer.GMTSaPromptMuons,
    GMTSaDisplacedMuons = l1tGTProducer.GMTSaDisplacedMuons,
    GMTTkMuons = l1tGTProducer.GMTTkMuons,
    CL2JetsSC4 = l1tGTProducer.CL2JetsSC4,
    CL2JetsSC8 = l1tGTProducer.CL2JetsSC8,
    CL2Electrons = l1tGTProducer.CL2Electrons,
    CL2Photons = l1tGTProducer.CL2Photons,
    CL2Taus = l1tGTProducer.CL2Taus,
    CL2EtSum = l1tGTProducer.CL2EtSum,
    CL2HtSum = l1tGTProducer.CL2HtSum,
    filename = cms.string("inputPattern"),
    bufferFileType = cms.string("input")
)

BoardDataOutputObjects = cms.EDAnalyzer("L1GTObjectBoardWriter",
    GTTPromptJets = l1tGTProducer.GTTPromptJets,
    GTTDisplacedJets = l1tGTProducer.GTTDisplacedJets,
    GTTPromptHtSum = l1tGTProducer.GTTPromptHtSum,
    GTTDisplacedHtSum = l1tGTProducer.GTTDisplacedHtSum,
    GTTEtSum = l1tGTProducer.GTTEtSum,
    GTTPrimaryVert = l1tGTProducer.GTTPrimaryVert,
    GMTSaPromptMuons = l1tGTProducer.GMTSaPromptMuons,
    GMTSaDisplacedMuons = l1tGTProducer.GMTSaDisplacedMuons,
    GMTTkMuons = l1tGTProducer.GMTTkMuons,
    CL2JetsSC4 = l1tGTProducer.CL2JetsSC4,
    CL2JetsSC8 = l1tGTProducer.CL2JetsSC8,
    CL2Electrons = l1tGTProducer.CL2Electrons,
    CL2Photons = l1tGTProducer.CL2Photons,
    CL2Taus = l1tGTProducer.CL2Taus,
    CL2EtSum = l1tGTProducer.CL2EtSum,
    CL2HtSum = l1tGTProducer.CL2HtSum,
    filename = cms.string("outputObjectPattern"),
    bufferFileType = cms.string("output")
)

AlgoBitBoardData = cms.EDAnalyzer("L1GTAlgoBoardWriter",
    outputFilename = cms.string("algoBitPattern"),
    algoBlocksTag = cms.InputTag("l1tGTAlgoBlockProducer"),
)
