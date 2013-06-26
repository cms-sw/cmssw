import FWCore.ParameterSet.Config as cms

SiStripSpyDisplay = cms.EDAnalyzer(
    'SiStripSpyDisplayModule',
    # Vector detIDs to examine
    detIDs = cms.vuint32( 369120277, 369120278 ),
    #
    # Spy Channel (raw) digi sources (change in your config)
    #========================================================
    # --- Source of (Spy) Scope Mode Raw Digis
    InputScopeModeRawDigiLabel = cms.InputTag("", ""),
    # --- Source of Virgin Raw digis
    InputPayloadRawDigiLabel = cms.InputTag("", ""),
    # --- Source of Processed Raw digis
    InputReorderedPayloadRawDigiLabel = cms.InputTag("", ""),
    # --- Source of Processed Raw digis
    InputReorderedModuleRawDigiLabel = cms.InputTag("", ""),
    # --- Source of Pedestals and Post-Pedestal digis
    InputPedestalsLabel           = cms.InputTag("", ""),
    InputNoisesLabel           = cms.InputTag("", ""),
    InputPostPedestalRawDigiLabel = cms.InputTag("", ""),
    # --- Source of Post-Common Mode digis
    InputPostCMRawDigiLabel = cms.InputTag("", ""),
    # --- Source of zero-suppressed raw digis
    InputZeroSuppressedRawDigiLabel = cms.InputTag("", ""),
    # --- Source of Zero-suppressed digis
    InputZeroSuppressedDigiLabel = cms.InputTag("", ""),
    # --- Mainline data for comparison
    InputCompVirginRawDigiLabel = cms.InputTag("", ""),
    InputCompZeroSuppressedDigiLabel = cms.InputTag("", ""),
    #
    # --- Folder name for TFileService output
    OutputFolderName = cms.string("DEFAULT_OUTPUTNAME"),
    #
    )
