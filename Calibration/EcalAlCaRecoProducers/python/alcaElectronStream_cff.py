import FWCore.ParameterSet.Config as cms

triggerStreamResultsFilter = cms.EDFilter('TriggerResultsFilter',
 hltResults = cms.InputTag('TriggerResults'), # HLT results - set to empty to ignore HLT                                                                                                 
 l1tResults = cms.InputTag(''), # L1 GT results - set to empty to ignore L1                                                                                                               
 l1tIgnoreMask = cms.bool(False), # use L1 mask                                                                                                                                           
 l1techIgnorePrescales = cms.bool(False), # read L1 technical bits from PSB#9, bypassing the prescales                                                                                    
 daqPartitions = cms.uint32(0x01), # used by the definition of the L1 mask                                                                                                                
 throw = cms.bool(True), # throw exception on unknown trigger names                                                                                                                       
 triggerConditions = cms.vstring(
  'HLT_Ele27_eta2p1_WP85_Gsf_v1',
  'HLT_Ele27_eta2p1_WP85_PFMET_MT50_Gsf_v1',
 )
)



HLTselectedElectronFEDList = cms.EDProducer("selectedElectronFEDListProducerGsf",
    dumpSelectedEcalFed = cms.bool(True),
    dEtaPixelRegion = cms.double(0.3),
    outputLabelModule = cms.string('StreamElectronRawFed'),
    recoEcalCandidateCollections = cms.VInputTag("hltEle27eta2p1WP85PFMT50PFMTFilter","hltEle27WP85GsfTrackIsoFilter"),
    addThisSelectedFEDs = cms.vint32(812,813),
    ESLookupTable = cms.string('EventFilter/ESDigiToRaw/data/ES_lookup_table.dat'),
    dRStripRegion = cms.double(0.3),
    HBHERecHitCollection = cms.InputTag("hltParticleFlowRecHitHCALForEgamma"),
    dumpSelectedSiStripFed = cms.bool(True),
    maxZPixelRegion = cms.double(24.0),
    dRHcalRegion = cms.double(0.5),
    dumpAllHcalFed = cms.bool(False),
    debug = cms.bool(False),
    electronCollections = cms.VInputTag("hltEgammaGsfElectrons"),
    dumpSelectedHCALFed = cms.bool(True),
    HCALLookUpTable = cms.string('Calibration/EcalAlCaRecoProducers/python/HcalElectronicsMap_v7.00_offline'),
    beamSpot = cms.InputTag("hltOnlineBeamSpot"),
    isGsfElectronCollection = cms.vint32(True),
    dPhiPixelRegion = cms.double(0.3),
    dumpSelectedSiPixelFed = cms.bool(True),
    dumpAllTrackerFed = cms.bool(False),
    rawDataLabel = cms.InputTag("rawDataCollector"),
    dumpAllEcalFed = cms.bool(False)
)


streamEvents = cms.untracked.PSet(
    SelectEvents   = cms.vstring('HLT_Ele27_eta2p1_WP85_Gsf_v1', 'HLT_Ele27_eta2p1_WP85_PFMET_MT50_Gsf_v1'),    
    outputCommands = cms.untracked.vstring('drop * ',
        'keep edmTriggerResults_*_*_*',
        'keep *_hltL1GtObjectMap_*_*',
        'drop *_*_*_*SIM*',
        'keep *_HLTselectedElectronFEDList_*StreamElectronRawFed*_*',
        'keep *_*hltFixedGridRhoFastjetAllCaloForMuons*_*_*',
        'keep *_*hltFixedGridRhoFastjetAll_*_*',
        'keep *_*hltPixelVerticesElectrons*_*_*',
        'keep *_*hltPixelVertices_*_*',
        'keep *_*hltPFMETProducer_*_*')
)



