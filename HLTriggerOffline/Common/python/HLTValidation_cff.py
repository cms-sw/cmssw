from Validation.RecoTrack.HLTmultiTrackValidator_cff import *
from Validation.RecoVertex.HLTmultiPVvalidator_cff import *
from HLTriggerOffline.Muon.HLTMuonVal_cff import *
from HLTriggerOffline.Tau.Validation.HLTTauValidation_cff import *
from HLTriggerOffline.Egamma.EgammaValidationAutoConf_cff import *
from HLTriggerOffline.HeavyFlavor.heavyFlavorValidationSequence_cff import *
from HLTriggerOffline.JetMET.Validation.HLTJetMETValidation_cff import *
#from HLTriggerOffline.special.hltAlCaVal_cff import *
from HLTriggerOffline.SUSYBSM.SusyExoValidation_cff import *
from HLTriggerOffline.Higgs.HiggsValidation_cff import *
from HLTriggerOffline.B2G.b2gHLTValidation_cff import *
from HLTriggerOffline.Exotica.ExoticaValidation_cff import *
from HLTriggerOffline.SMP.SMPValidation_cff import *
from HLTriggerOffline.Btag.HltBtagValidation_cff import *
from HLTriggerOffline.Egamma.HLTmultiTrackValidatorGsfTracks_cff import *
# HCAL
from Validation.HcalDigis.HLTHcalDigisParam_cfi import *
from Validation.HcalRecHits.HLTHcalRecHitParam_cfi import *
## SiTracker Phase2
from Validation.SiTrackerPhase2V.HLTPhase2TrackerValidationFirstStep_cff import *
# Gen-level Validation
from Validation.HLTrigger.HLTGenValidation_cff import *
from Validation.RecoParticleFlow.DQMForPF_MiniAOD_cff import *
from Validation.Configuration.globalValidation_cff import *

# HGCAL Rechit Calibration
from Validation.HGCalValidation.hgcalHitCalibrationDefault_cfi import hgcalHitCalibrationDefault as _hgcalHitCalibrationDefault
hgcalHitCalibrationHLT = _hgcalHitCalibrationDefault.clone(
    folder = "HLT/HGCalHitCalibration",
    recHitsEE = ("hltHGCalRecHit", "HGCEERecHits", "HLT"),
    recHitsFH = ("hltHGCalRecHit", "HGCHEFRecHits", "HLT"),
    recHitsBH = ("hltHGCalRecHit", "HGCHEBRecHits", "HLT"),
    hgcalMultiClusters = "None",
    electrons = "None",
    photons = "None"
)

# HGCAL validation
from Validation.HGCalValidation.HLTHGCalValidator_cff import *
from RecoHGCal.TICL.HLTSimTracksters_cff import *

# offline dqm:
# from DQMOffline.Trigger.DQMOffline_Trigger_cff.py import *
from DQMOffline.Trigger.HLTTauDQMOffline_cff import *
from DQMOffline.Trigger.EgHLTOfflineSource_cfi import *
#from DQMOffline.Trigger.MuonTrigRateAnalyzer_cfi import *

# online dqm:
from DQMOffline.Trigger.HLTMonTau_cfi import *

# additional producer sequence prior to hltvalidation
# to evacuate producers/filters from the EndPath
hltassociation = cms.Sequence(
    hltMultiTrackValidation
    +hltMultiPVValidation
    +egammaSelectors
    +ExoticaValidationProdSeq
    +hltMultiTrackValidationGsfTracks
    +hltJetPreValidSeq
    )
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel

# Temporary Phase-2 config
from Configuration.Eras.Modifier_phase2_common_cff import phase2_common

# Create the modified sequence for phase 2
_phase2_hltassociation = hltassociation.copyAndExclude([
    egammaSelectors,
    ExoticaValidationProdSeq,
])

# Add hltTrackerphase2ValidationSource to the sequence
_phase2_hltassociation += hltTrackerphase2ValidationSource

# Add HGCal SimTracksters
_phase2_hltassociation += hltTiclSimTrackstersSeq

# Apply the modification
phase2_common.toReplaceWith(hltassociation, _phase2_hltassociation)

# hcal
from DQMOffline.Trigger.HCALMonitoring_cff import *

hltvalidationCommon = cms.Sequence(
    hcalMonitoringSequence
)

hltvalidationWithMC = cms.Sequence(
    HLTMuonVal
    +HLTTauVal
    +egammaValidationSequence
    +heavyFlavorValidationSequence
    +HLTJetMETValSeq
    +HLTSusyExoValSeq
    +HiggsValidationSequence
    +ExoticaValidationSequence
    +b2gHLTriggerValidation
    +SMPValidationSequence
    +hltbtagValidationSequence #too noisy for now
    +hltHCALdigisAnalyzer+hltHCALRecoAnalyzer+hltHCALNoiseRates # HCAL
)

# Exclude everything except Muon and JetMET for now. Add HGCAL Hit Calibration
_hltvalidationWithMC_Phase2 = hltvalidationWithMC.copyAndExclude([#HLTMuonVal,
  HLTTauVal,
  egammaValidationSequence,
  heavyFlavorValidationSequence,
  #HLTJetMETValSeq,
  HLTSusyExoValSeq,
  HiggsValidationSequence,
  ExoticaValidationSequence,
  b2gHLTriggerValidation,
  SMPValidationSequence,
  hltbtagValidationSequence,
  hltHCALdigisAnalyzer,
  hltHCALRecoAnalyzer,
  hltHCALNoiseRates])
_hltvalidationWithMC_Phase2.insert(-1, hgcalHitCalibrationHLT)
_hltvalidationWithMC_Phase2.insert(-1, hltHgcalValidator)
_hltvalidationWithMC_Phase2.insert(-1, hltGENValidation)
phase2_common.toReplaceWith(hltvalidationWithMC, _hltvalidationWithMC_Phase2)

hltvalidationWithData = cms.Sequence(
)

hltvalidation = cms.Sequence(
    hltvalidationCommon *
    hltvalidationWithMC *
    hltvalidationWithData
)

from RecoLocalCalo.HGCalRecProducers.recHitMapProducer_cff import recHitMapProducer as _recHitMapProducer
hltRecHitMapProducer = _recHitMapProducer.clone()

ticl_barrel.toModify(hltRecHitMapProducer,
                     hits = ["hltHGCalRecHit:HGCEERecHits", "hltHGCalRecHit:HGCHEFRecHits", "hltHGCalRecHit:HGCHEBRecHits",
                             "hltParticleFlowRecHitECALUnseeded", "hltParticleFlowRecHitHBHE"],
                     hgcalOnly = False
                     )

# ["hltHGCalRecHit:HGCEERecHits", "hltHGCalRecHit:HGCHEFRecHits", "hltHGCalRecHit:HGCHEBRecHits"]
# ["hltParticleFlowRecHitECALUnseeded", "hltParticleFlowRecHitHBHE"]
hlthits_hgcal = cms.InputTag("hltRecHitMapProducer", "RefProdVectorHGCRecHitCollection")
hlthits_barrel = cms.InputTag("hltRecHitMapProducer", "RefProdVectorPFRecHitCollection")

# adapt associators for HLT Barrel
assoc_barrel_args = dict(
    hitMapTag = cms.InputTag('hltRecHitMapProducer', 'barrelRecHitMap'),
    hits = hlthits_barrel,
)
assoc_hgcal_args = dict(
    hitMapTag = cms.InputTag('hltRecHitMapProducer', 'hgcalRecHitMap'),
    hits = hlthits_hgcal,
)

from SimCalorimetry.HGCalAssociatorProducers.barrelLCToSCAssociatorByEnergyScoreProducer_cfi import *
from SimCalorimetry.HGCalAssociatorProducers.barrelLCToCPAssociatorByEnergyScoreProducer_cfi import *
hltBarrelLCToCPAssociatorByEnergyScoreProducer = barrelLCToCPAssociatorByEnergyScoreProducer.clone(**assoc_barrel_args)
hltBarrelLCToSCAssociatorByEnergyScoreProducer = barrelLCToSCAssociatorByEnergyScoreProducer.clone(**assoc_barrel_args)

from SimCalorimetry.HGCalSimProducers.hgcHitAssociation_cfi import lcAssocByEnergyScoreProducer, scAssocByEnergyScoreProducer
hltHGCalLCToCPAssociatorByEnergyScoreProducer = lcAssocByEnergyScoreProducer.clone(**assoc_hgcal_args)
hltHGCalLCToSCAssociatorByEnergyScoreProducer = scAssocByEnergyScoreProducer.clone(**assoc_hgcal_args)

from SimCalorimetry.HGCalAssociatorProducers.LCToCPAssociation_cfi import barrelLayerClusterCaloParticleAssociation as _barrelLayerClusterCaloParticleAssociation
hltBarrelLayerClusterCaloParticleAssociation = _barrelLayerClusterCaloParticleAssociation.clone(
    associator = cms.InputTag('hltBarrelLCToCPAssociatorByEnergyScoreProducer'),
    label_lc = cms.InputTag('hltMergeLayerClusters')
)
from SimCalorimetry.HGCalAssociatorProducers.LCToCPAssociation_cfi import layerClusterCaloParticleAssociation as _layerClusterCaloParticleAssociation
hltHGCalLayerClusterCaloParticleAssociation = _layerClusterCaloParticleAssociation.clone(
    associator = cms.InputTag('hltHGCalLCToCPAssociatorByEnergyScoreProducer'),
    label_lc = cms.InputTag('hltMergeLayerClusters')
)

from SimCalorimetry.HGCalAssociatorProducers.LCToSCAssociation_cfi import barrelLayerClusterSimClusterAssociation as _barrelLayerClusterSimClusterAssociation
hltBarrelLayerClusterSimClusterAssociation = _barrelLayerClusterSimClusterAssociation.clone(
    associator = cms.InputTag('hltBarrelLCToSCAssociatorByEnergyScoreProducer'),
    label_lcl = cms.InputTag('hltMergeLayerClusters')
)
from SimCalorimetry.HGCalAssociatorProducers.LCToSCAssociation_cfi import layerClusterSimClusterAssociation as _layerClusterSimClusterAssociation
hltHGCalLayerClusterSimClusterAssociation = _layerClusterSimClusterAssociation.clone(
    associator = cms.InputTag('hltHGCalLCToSCAssociatorByEnergyScoreProducer'),
    label_lcl = cms.InputTag('hltMergeLayerClusters')
)

from Validation.HGCalValidation.barrelValidator_cfi import barrelValidator as _barrelValidator

# from Configuration.StandardSequences.Validation_cff import prevalidation
# ImportError: cannot import name 'prevalidation' from partially initialized module 'Configuration.StandardSequences.Validation_cff' (most likely due to a circular import) (/shared/CMSSW_15_1_X_2025-07-16-2300/src/Configuration/StandardSequences/python/Validation_cff.py)
hltprevalidation = cms.Sequence( cms.SequencePlaceholder("mix") * globalPrevalidation * hltassociation * metPreValidSeq * jetPreValidSeq )
phase2_common.toReplaceWith(hltprevalidation, hltprevalidation.copyAndExclude([cms.SequencePlaceholder("mix"),globalPrevalidation,metPreValidSeq,jetPreValidSeq]))

_hltprevalidation_Phase2 = hltprevalidation.copy()
_hltprevalidation_Phase2.insert(
    -1,
    cms.Sequence(
        hltRecHitMapProducer *
        hltHGCalLCToCPAssociatorByEnergyScoreProducer *
        hltHGCalLCToSCAssociatorByEnergyScoreProducer *
        hltHGCalLayerClusterCaloParticleAssociation *
        hltHGCalLayerClusterSimClusterAssociation
    )
    )

phase2_common.toReplaceWith(hltprevalidation, _hltprevalidation_Phase2)

_hltprevalidation_Phase2_WithBarrel = _hltprevalidation_Phase2.copy()
_hltprevalidation_Phase2_WithBarrel.insert(
    -1,
    cms.Sequence(
        hltBarrelLCToCPAssociatorByEnergyScoreProducer *
        hltBarrelLCToSCAssociatorByEnergyScoreProducer *
        hltBarrelLayerClusterCaloParticleAssociation *
        hltBarrelLayerClusterSimClusterAssociation
    )
)
ticl_barrel.toReplaceWith(hltprevalidation, _hltprevalidation_Phase2_WithBarrel)

hltvalidationCommon = hltvalidationCommon.copy()
hltvalidationWithMC = hltvalidationWithMC.copy()
hltvalidationWithData = hltvalidationWithData.copy()
_hltvalidationWithMC_Phase2 = _hltvalidationWithMC_Phase2.copy()
phase2_common.toReplaceWith(hltvalidationWithMC, _hltvalidationWithMC_Phase2)

from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel
_hltvalidationWithMC_Phase2_WithBarrel = _hltvalidationWithMC_Phase2.copy()

hltbarrelvalidation = _barrelValidator.clone(
    lclTag = "hltMergeLayerClusters",
    hits = hlthits_barrel,
    rechitmapTag = cms.InputTag("hltRecHitMapProducer", "barrelRecHitMap"),
    associator = ['hltBarrelLayerClusterCaloParticleAssociation',],
    associatorSim = ['hltBarrelLayerClusterSimClusterAssociation',],
    dirName = 'HLT/BarrelCalorimeters/BarrelValidator/'
)
_hltvalidationWithMC_Phase2_WithBarrel.insert(-1, hltbarrelvalidation)
ticl_barrel.toReplaceWith(hltvalidationWithMC, _hltvalidationWithMC_Phase2_WithBarrel)

# hgcalLocalRecoTask = cms.Task( HGCalUncalibRecHit,
#                                        HGCalRecHit,
#                                        recHitMapProducer,

hltvalidation = cms.Sequence(
    hltvalidationCommon * # HCAL RecHit analyzer
    hltvalidationWithMC *
    hltvalidationWithData *
    DQMHLTPF
)

# some hlt collections have no direct fastsim equivalent
# remove the dependent modules for now
# probably it would be rather easy to add or fake these collections
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toReplaceWith(hltassociation, hltassociation.copyAndExclude([
    hltMultiTrackValidation,
    hltMultiPVValidation,
    hltMultiTrackValidationGsfTracks
]))

from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
for e in [pp_on_XeXe_2017, pp_on_AA]:
    e.toReplaceWith(hltvalidation, hltvalidation.copyAndExclude([HiggsValidationSequence]))

hltvalidation_preprod = cms.Sequence(
  HLTTauVal
  +heavyFlavorValidationSequence
  +HLTSusyExoValSeq
# +HiggsValidationSequence
)

hltvalidation_prod = cms.Sequence(
)

trigdqm_forValidation = cms.Sequence(
    hltMonTauReco+HLTTauDQMOffline
    +egHLTOffDQMSource
)

hltvalidation_withDQM = cms.Sequence(
    hltvalidation
    +trigdqm_forValidation
)

    
