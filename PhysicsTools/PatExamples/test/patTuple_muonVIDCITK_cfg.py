from PhysicsTools.PatAlgos.patTemplate_cfg import *
process.options.allowUnscheduled = cms.untracked.bool(True)
#process.Tracer = cms.Service("Tracer")

process.load("PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff")
process.load("PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff")

process.load("RecoMuon.MuonIdentification.Identification.cutBasedMuonId_MuonPOG_V0_cff")
process.load("RecoMuon.MuonIsolation.muonPFIsolationCitk_cff")
from PhysicsTools.SelectorUtils.tools.vid_id_tools import *
switchOnVIDMuonIdProducer(process, DataFormat.AOD)
setupVIDMuonSelection(process, process.cutBasedMuonId_MuonPOG_V0_loose)
setupVIDMuonSelection(process, process.cutBasedMuonId_MuonPOG_V0_medium)
setupVIDMuonSelection(process, process.cutBasedMuonId_MuonPOG_V0_tight)
setupVIDMuonSelection(process, process.cutBasedMuonId_MuonPOG_V0_soft)
setupVIDMuonSelection(process, process.cutBasedMuonId_MuonPOG_V0_highpt)
process.muoMuonIDs.physicsObjectSrc = "patMuons"
process.muonPFNoPileUpIsolation.srcToIsolate = "patMuons"
process.muonPFPileUpIsolation.srcToIsolate = "patMuons"

#   process.GlobalTag.globaltag =  ...    ##  (according to https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions)
from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValProdTTbarAODSIM
process.source.fileNames = filesRelValProdTTbarAODSIM
process.maxEvents.input = -1
process.out.fileName = 'patTuple_isoval.root'
process.options.wantSummary = False

process.muonVIDCITKAnalyzer = cms.EDAnalyzer("MuonVIDCITKAnalyzer",
    muon = cms.InputTag("patMuons"),
    vertex = cms.InputTag("offlinePrimaryVertices"),
)

process.p = cms.Path(
    process.muonVIDCITKAnalyzer
)

process.pfPileUpIso.PFCandidates = 'particleFlowPtrs'
process.pfNoPileUpIso.bottomCollection = 'particleFlowPtrs'
process.out.outputCommands = [
    'drop *',
    'keep recoMuons_muons_*_*',
    'keep *_patMuons_*_*',
    'keep *_muPFIsoValue*PAT_*_*',
    'keep *_muoMuonIDs_*_*',
    'keep *_muons_muPFSumDRIsoValue*04_*', # Standard IsoDeposit in RECO
    'keep *_muonPFNoPileUpIsolation_*_*', # Isolation from CITK
    'keep *_muonPFPileUpIsolation_*_*', # Isolation from CITK
]
from PhysicsTools.PatAlgos.patEventContent_cff import patEventContentNoCleaning
from PhysicsTools.PatAlgos.patEventContent_cff import patExtraAodEventContent
process.out.outputCommands += patEventContentNoCleaning
process.out.outputCommands += patExtraAodEventContent

process.out.SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('p'))
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(10000)
process.TFileService = cms.Service("TFileService",
  fileName = cms.string('TFileServiceOutput.root')
)

