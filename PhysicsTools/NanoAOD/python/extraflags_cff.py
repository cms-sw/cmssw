import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.nano_eras_cff import *

# Bad/clone muon filters - tagging mode to keep the event
from RecoMET.METFilters.badGlobalMuonTaggersMiniAOD_cff import badGlobalMuonTaggerMAOD, cloneGlobalMuonTaggerMAOD
badGlobalMuonTagger = badGlobalMuonTaggerMAOD.clone(
    taggingMode = True
)

cloneGlobalMuonTagger = cloneGlobalMuonTaggerMAOD.clone(
    taggingMode = True
)

from RecoMET.METFilters.BadPFMuonFilter_cfi import BadPFMuonFilter
BadPFMuonTagger = BadPFMuonFilter.clone(
    PFCandidates = cms.InputTag("packedPFCandidates"),
    muons = cms.InputTag("slimmedMuons"),
    vtx = cms.InputTag("offlineSlimmedPrimaryVertices"),
    taggingMode = True,
)

# Bad charge hadron
from RecoMET.METFilters.BadChargedCandidateFilter_cfi import BadChargedCandidateFilter
BadChargedCandidateTagger = BadChargedCandidateFilter.clone(
    PFCandidates = cms.InputTag("packedPFCandidates"),
    muons = cms.InputTag("slimmedMuons"),
    vtx = cms.InputTag("offlineSlimmedPrimaryVertices"),
    taggingMode = True,
)

extraFlagsTable = cms.EDProducer("GlobalVariablesTableProducer",
    variables = cms.PSet(
        Flag_BadGlobalMuon = ExtVar(cms.InputTag("badGlobalMuonTagger:notBadEvent"), bool, doc = "Bad muon flag"),
        Flag_CloneGlobalMuon = ExtVar(cms.InputTag("cloneGlobalMuonTagger:notBadEvent"), bool, doc = "Clone muon flag"),
        Flag_BadPFMuonFilter = ExtVar(cms.InputTag("BadPFMuonTagger"), bool, doc = "Bad PF muon flag"),
        Flag_BadChargedCandidateFilter = ExtVar(cms.InputTag("BadChargedCandidateTagger"), bool, doc = "Bad charged hadron flag"),
    )
)

from RecoMET.METFilters.ecalBadCalibFilter_cfi import *
ecalBadCalibFilterNanoTagger = ecalBadCalibFilter.clone(
    taggingMode = cms.bool(True)
)

# modify extraFlagsTable to store ecalBadCalibFilter decision which is re-run with updated bad crystal list for 2017 and 2018 samples
for modifier in run2_nanoAOD_94XMiniAODv1, run2_nanoAOD_94XMiniAODv2, run2_nanoAOD_102Xv1:
    modifier.toModify(extraFlagsTable, variables= cms.PSet())
    modifier.toModify(extraFlagsTable, variables = dict(Flag_ecalBadCalibFilterV2 = ExtVar(cms.InputTag("ecalBadCalibFilterNanoTagger"), bool, doc = "Bad ECAL calib flag (updatedxtal list)")))


# empty task as default
extraFlagsProducersTask = cms.Task()
extraFlagsTableTask = cms.Task()

## those need to be added only for some specific eras
extraFlagsProducersTask_80x = cms.Task(badGlobalMuonTagger,cloneGlobalMuonTagger,BadPFMuonTagger,BadChargedCandidateTagger)
extraFlagsProducersTask_102x = cms.Task(ecalBadCalibFilterNanoTagger)

run2_miniAOD_80XLegacy.toReplaceWith(extraFlagsProducersTask, extraFlagsProducersTask_80x)
run2_miniAOD_80XLegacy.toReplaceWith(extraFlagsTableTask, cms.Task(extraFlagsTable))

(run2_nanoAOD_94XMiniAODv1 | run2_nanoAOD_94XMiniAODv2 | run2_nanoAOD_102Xv1).toReplaceWith(extraFlagsProducersTask, extraFlagsProducersTask_102x)
(run2_nanoAOD_94XMiniAODv1 | run2_nanoAOD_94XMiniAODv2 | run2_nanoAOD_102Xv1).toReplaceWith(extraFlagsTableTask, cms.Task(extraFlagsTable))
