import FWCore.ParameterSet.Config as cms

## We don't use "import *" because the cff contains some modules for which the C++ class doesn't exist
## and this triggers an error under unscheduled mode
from RecoMET.METFilters.metFilters_cff import HBHENoiseFilterResultProducer, HBHENoiseFilter, HBHENoiseIsoFilter, hcalLaserEventFilter
from RecoMET.METFilters.metFilters_cff import EcalDeadCellTriggerPrimitiveFilter, eeBadScFilter, ecalLaserCorrFilter, EcalDeadCellBoundaryEnergyFilter, ecalBadCalibFilter
from RecoMET.METFilters.metFilters_cff import primaryVertexFilter, CSCTightHaloFilter, CSCTightHaloTrkMuUnvetoFilter, CSCTightHalo2015Filter, globalTightHalo2016Filter, globalSuperTightHalo2016Filter, HcalStripHaloFilter
from RecoMET.METFilters.metFilters_cff import goodVertices, trackingFailureFilter, trkPOGFilters, manystripclus53X, toomanystripclus53X, logErrorTooManyClusters
from RecoMET.METFilters.metFilters_cff import chargedHadronTrackResolutionFilter, muonBadTrackFilter
from RecoMET.METFilters.metFilters_cff import BadChargedCandidateFilter, BadPFMuonFilter #2016 post-ICHEPversion
from RecoMET.METFilters.metFilters_cff import BadChargedCandidateSummer16Filter, BadPFMuonSummer16Filter #2016 ICHEP version
from RecoMET.METFilters.metFilters_cff import metFilters

# individual filters
Flag_HBHENoiseFilter = cms.Path(HBHENoiseFilterResultProducer * HBHENoiseFilter)
Flag_HBHENoiseIsoFilter = cms.Path(HBHENoiseFilterResultProducer * HBHENoiseIsoFilter)
Flag_CSCTightHaloFilter = cms.Path(CSCTightHaloFilter)
Flag_CSCTightHaloTrkMuUnvetoFilter = cms.Path(CSCTightHaloTrkMuUnvetoFilter)
Flag_CSCTightHalo2015Filter = cms.Path(CSCTightHalo2015Filter)
Flag_globalTightHalo2016Filter = cms.Path(globalTightHalo2016Filter)
Flag_globalSuperTightHalo2016Filter = cms.Path(globalSuperTightHalo2016Filter)
Flag_HcalStripHaloFilter = cms.Path(HcalStripHaloFilter)
Flag_hcalLaserEventFilter = cms.Path(hcalLaserEventFilter)
Flag_EcalDeadCellTriggerPrimitiveFilter = cms.Path(EcalDeadCellTriggerPrimitiveFilter)
Flag_EcalDeadCellBoundaryEnergyFilter = cms.Path(EcalDeadCellBoundaryEnergyFilter)
Flag_ecalBadCalibFilter = cms.Path()
Flag_goodVertices = cms.Path(primaryVertexFilter)
Flag_trackingFailureFilter = cms.Path(goodVertices + trackingFailureFilter)
Flag_eeBadScFilter = cms.Path(eeBadScFilter)
Flag_ecalLaserCorrFilter = cms.Path(ecalLaserCorrFilter)
Flag_trkPOGFilters = cms.Path(trkPOGFilters)
Flag_chargedHadronTrackResolutionFilter = cms.Path(chargedHadronTrackResolutionFilter)
Flag_muonBadTrackFilter = cms.Path(muonBadTrackFilter)
Flag_BadChargedCandidateFilter = cms.Path(BadChargedCandidateFilter)
Flag_BadPFMuonFilter = cms.Path(BadPFMuonFilter)
Flag_BadChargedCandidateSummer16Filter = cms.Path(BadChargedCandidateSummer16Filter)
Flag_BadPFMuonSummer16Filter = cms.Path(BadPFMuonSummer16Filter)

# and the sub-filters
Flag_trkPOG_manystripclus53X = cms.Path(~manystripclus53X)
Flag_trkPOG_toomanystripclus53X = cms.Path(~toomanystripclus53X)
Flag_trkPOG_logErrorTooManyClusters = cms.Path(~logErrorTooManyClusters)


# and the summary
Flag_METFilters = cms.Path(metFilters)

#add your new path here!!
allMetFilterPaths=['HBHENoiseFilter','HBHENoiseIsoFilter','CSCTightHaloFilter','CSCTightHaloTrkMuUnvetoFilter','CSCTightHalo2015Filter','globalTightHalo2016Filter','globalSuperTightHalo2016Filter','HcalStripHaloFilter','hcalLaserEventFilter','EcalDeadCellTriggerPrimitiveFilter','EcalDeadCellBoundaryEnergyFilter','ecalBadCalibFilter','goodVertices','eeBadScFilter',
                   'ecalLaserCorrFilter','trkPOGFilters','chargedHadronTrackResolutionFilter','muonBadTrackFilter',
                   'BadChargedCandidateFilter','BadPFMuonFilter','BadChargedCandidateSummer16Filter','BadPFMuonSummer16Filter',
                   'trkPOG_manystripclus53X','trkPOG_toomanystripclus53X','trkPOG_logErrorTooManyClusters','METFilters']

       
def miniAOD_customizeMETFiltersFastSim(process):
    """Replace some MET filters that don't work in FastSim with trivial bools"""
    for X in 'CSCTightHaloFilter', 'CSCTightHaloTrkMuUnvetoFilter','CSCTightHalo2015Filter','globalTightHalo2016Filter','globalSuperTightHalo2016Filter','HcalStripHaloFilter':
        process.globalReplace(X, cms.EDFilter("HLTBool", result=cms.bool(True)))    
    for X in 'manystripclus53X', 'toomanystripclus53X', 'logErrorTooManyClusters':
        process.globalReplace(X, cms.EDFilter("HLTBool", result=cms.bool(False)))
    return process

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toReplaceWith( Flag_trkPOG_manystripclus53X, cms.Path() )
phase2_common.toReplaceWith( Flag_trkPOG_toomanystripclus53X, cms.Path() )
phase2_common.toReplaceWith( Flag_trkPOGFilters, cms.Path(~logErrorTooManyClusters) )

from Configuration.Eras.Modifier_run2_common_cff import run2_common
run2_common.toReplaceWith( Flag_ecalBadCalibFilter, cms.Path(ecalBadCalibFilter) )

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toReplaceWith( Flag_HBHENoiseFilter, cms.Path() )
phase2_hgcal.toReplaceWith( Flag_HBHENoiseIsoFilter, cms.Path() )
phase2_hgcal.toReplaceWith( Flag_eeBadScFilter, cms.Path() )

metFilterPathsTask = cms.Task(
    HBHENoiseFilterResultProducer,
    HBHENoiseFilter,
    HBHENoiseIsoFilter,
    hcalLaserEventFilter,
    EcalDeadCellTriggerPrimitiveFilter,
    eeBadScFilter,
    ecalLaserCorrFilter,
    EcalDeadCellBoundaryEnergyFilter,
    ecalBadCalibFilter,
    primaryVertexFilter,
    CSCTightHaloFilter,
    CSCTightHaloTrkMuUnvetoFilter,
    CSCTightHalo2015Filter,
    globalTightHalo2016Filter,
    globalSuperTightHalo2016Filter,
    HcalStripHaloFilter,
    goodVertices,
    trackingFailureFilter,
    manystripclus53X,
    toomanystripclus53X,
    logErrorTooManyClusters,
    chargedHadronTrackResolutionFilter,
    muonBadTrackFilter,
    BadChargedCandidateFilter,
    BadPFMuonFilter,
    BadChargedCandidateSummer16Filter,
    BadPFMuonSummer16Filter
)
phase2_common.toReplaceWith( metFilterPathsTask, metFilterPathsTask.copyAndExclude( [ manystripclus53X, toomanystripclus53X ] ) )
phase2_hgcal.toReplaceWith( metFilterPathsTask, metFilterPathsTask.copyAndExclude( [ HBHENoiseFilterResultProducer, HBHENoiseFilter, HBHENoiseIsoFilter, eeBadScFilter ] ) )
