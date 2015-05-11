## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *
## switch to uncheduled mode
process.options.allowUnscheduled = cms.untracked.bool(True)
#process.Tracer = cms.Service("Tracer")

process.load("PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff")
process.load("PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff")

process.load("RecoMuon.MuonIdentification.Identification.cutBasedMuonId_MuonPOG_V0_cff")
process.load("RecoMuon.MuonIsolation.muonPFIsolationCitk_cff")
from PhysicsTools.SelectorUtils.tools.vid_id_tools import *
switchOnVIDMuonIdProducer(process)
setupVIDMuonSelection(process, process.cutBasedMuonId_MuonPOG_V0)

## ------------------------------------------------------
#  In addition you usually want to change the following
#  parameters:
## ------------------------------------------------------
#
#   process.GlobalTag.globaltag =  ...    ##  (according to https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions)
#                                         ##
from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValProdTTbarAODSIM
process.source.fileNames = filesRelValProdTTbarAODSIM
#                                         ##
process.maxEvents.input = 2000
#                                         ##
#   process.out.outputCommands = [ ... ]  ##  (e.g. taken from PhysicsTools/PatAlgos/python/patEventContent_cff.py)
#                                         ##
process.out.fileName = 'patTuple_isoval.root'
#                                         ##
#   process.options.wantSummary = False   ##  (to suppress the long output at the end of the job)

process.out.outputCommands = [
    'keep *_muonPFNoPileUpIsolation_*_*',
    'keep *_muonPFPileUpIsolation_*_*',
#    'keep *_muPFIsoValue*_*_*',
    'keep recoMuons_muons_*_*',
    'keep *_muonVIDs_*_*',
    'keep *_patMuons_*_*',
]

