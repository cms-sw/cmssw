import FWCore.ParameterSet.Config as cms

SiStripO2OValidationParameters = cms.PSet(
    # Enable/Disable O2O of the following Objects
    ValidateFEDCabling=cms.untracked.bool(True),
    ValidateThreshold=cms.untracked.bool(True),
    ValidateQuality=cms.untracked.bool(True),
    ValidateNoise=cms.untracked.bool(True),
    ValidatePedestal=cms.untracked.bool(True),
    ValidateAPVLatency=cms.untracked.bool(True),
    ValidateAPVTiming=cms.untracked.bool(True),

    #Root file that will be created/taken by validation tool
    RootFile=cms.untracked.string("SiStripO2OValidation.root"),

    #Outputfile extension used by tracker map
    FileExtension=cms.untracked.string("jpg"),

    # Print additional debug information
    DebugMode=cms.untracked.bool(False)
    )

