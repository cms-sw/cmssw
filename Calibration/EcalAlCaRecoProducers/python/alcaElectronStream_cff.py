import FWCore.ParameterSet.Config as cms

triggerStreamResultsFilter = cms.EDFilter('TriggerResultsFilter',
 hltResults = cms.InputTag('TriggerResults'),   # HLT results - set to empty to ignore HLT
 l1tResults = cms.InputTag(''),                 # L1 uGT results - set to empty to ignore L1
 l1tIgnoreMaskAndPrescale = cms.bool(False),    # use L1 results before masks and prescales
 throw = cms.bool(True),                        # throw exception on unknown trigger names
 triggerConditions = cms.vstring(
  'HLT_Ele27_eta2p1_WP85_Gsf_v1',
  'HLT_Ele27_eta2p1_WP85_PFMET_MT50_Gsf_v1',
 )
)



HLTselectedElectronFEDList = cms.EDProducer("selectedElectronFEDListProducerGsf",
    recoEcalCandidateTags   = cms.VInputTag("hltEle27eta2p1WP85PFMT50PFMTFilter","hltEle27WP85GsfTrackIsoFilter"),
    electronTags            = cms.VInputTag("hltEgammaGsfElectrons"),
    isGsfElectronCollection = cms.vint32(True),
    beamSpotTag             = cms.InputTag("hltOnlineBeamSpot"),
    rawDataTag              = cms.InputTag("rawDataCollector"),
    HBHERecHitTag           = cms.InputTag("hltHbhereco"),
    ESLookupTable           = cms.string('EventFilter/ESDigiToRaw/data/ES_lookup_table.dat'),
    dumpSelectedEcalFed     = cms.bool(True),
    dumpSelectedSiPixelFed  = cms.bool(True),
    dumpSelectedSiStripFed  = cms.bool(True),
    dumpSelectedHCALFed     = cms.bool(True),
    dumpAllEcalFed          = cms.bool(False),
    dumpAllHcalFed          = cms.bool(False),
    dumpAllTrackerFed       = cms.bool(False),
    dPhiPixelRegion         = cms.double(0.3),
    dEtaPixelRegion         = cms.double(0.3),
    maxZPixelRegion         = cms.double(24.0),
    dRStripRegion           = cms.double(0.3),
    dRHcalRegion            = cms.double(0.5),
    outputLabelModule       = cms.string('StreamElectronRawFed'),
    addThisSelectedFEDs     = cms.vint32(812,813)
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



