import FWCore.ParameterSet.Config as cms

pandorapfanew = cms.EDProducer('PandoraCMSPFCandProducer',
    debugPrint = cms.bool(False), #for cout statements
    debugHisto = cms.bool(False), #for diagnostic/calibration histograms
    useRecoTrackAsssociation = cms.bool(False), #needed to turn off for 140PU                                                               
    HGCrechitCollection  = cms.InputTag("particleFlowRecHitHGCEE",""), 
    generaltracks = cms.InputTag("HGCalTrackCollection","TracksInHGCal"),
    tPRecoTrackAsssociation= cms.InputTag("trackingParticleRecoTrackAsssociation"),
    genParticles= cms.InputTag("genParticles"),
    # use slow algorithms until fast algoritms are available in the CMSSW external pandora library
#    inputconfigfile = cms.FileInPath('RecoParticleFlow/PandoraTranslator/data/PandoraSettingsBasic_cms_slow.xml'),
    inputconfigfile = cms.FileInPath('RecoParticleFlow/PandoraTranslator/data/PandoraSettingsBasic_cms.xml'),

    energyCorrMethod = cms.string('ABSCORR'),
#   absorber thickness correction
#   energyCorrMethod = cms.string('WEIGHTING'),
    energyWeightFile = cms.FileInPath('RecoParticleFlow/PandoraTranslator/data/energyWeight.txt'),

    calibrParFile = cms.FileInPath('RecoParticleFlow/PandoraTranslator/data/pandoraCalibrPars_pedro05032015.txt'),
    layerDepthFile = cms.FileInPath('RecoParticleFlow/PandoraTranslator/data/HGCmaterial_v5.root'),
    overburdenDepthFile = cms.FileInPath('RecoParticleFlow/PFClusterProducer/data/HGCMaterialOverburden.root'),
    useOverburdenCorrection = cms.bool(False), #disabled until the overburden values make sense
    pf_electron_output_col=cms.string('electrons'),
    outputFile = cms.string('pandoraoutput.root'),
    MaxDeltaPtOverPtForPfo = cms.double(1.00),
    MaxDeltaPtOverPtForClusterlessPfo = cms.double(0.50),
)
