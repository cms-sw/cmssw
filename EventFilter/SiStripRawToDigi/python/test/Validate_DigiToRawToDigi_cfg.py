import FWCore.ParameterSet.Config as cms

# The VR, PR and FK modes can only be used with the TRIV source
Source = str("SIM") # Options: "TRIV", "SIM"
Mode = str("ZS") # Options: "ZS", "VR", "PR", "FK"

if Source == str("SIM") and Mode != str("ZS") :
    print "The VR, PR and FK modes can only be used with the TRIV source!"
    import sys
    sys.exit()

process = cms.Process("DigiToRawToDigi")

process.load("EventFilter.SiStripRawToDigi.test.Validate_DigiToRawToDigi_cff")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )


# ----- InputSource -----


if Source == str("TRIV") :

    process.source = cms.Source(
        "EmptySource",
        firstRun = cms.untracked.uint32(999999)
        )

    process.SiStripDigiToRaw.InputModuleLabel = 'DigiSource'
    process.SiStripDigiToRaw.InputDigiLabel = ''

    process.DigiValidator.TagCollection1 = "DigiSource"
    process.newDigiValidator.TagCollection1 = "DigiSource"

    process.p = cms.Path( process.DigiSource * process.s )

elif Source == str("SIM") :

    process.source = cms.Source(
        "PoolSource",
        fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_1_0_pre2/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0001/027A5B5E-AF03-DE11-832C-000423D99AAE.root'
        )
        )
    
    process.SiStripDigiToRaw.InputModuleLabel = 'simSiStripDigis'
    process.SiStripDigiToRaw.InputDigiLabel = 'ZeroSuppressed'

    process.DigiValidator.TagCollection1 = "simSiStripDigis:ZeroSuppressed"
    process.newDigiValidator.TagCollection1 = "simSiStripDigis:ZeroSuppressed"

    process.p = cms.Path( process.s )

else :

    print "UNKNOWN INPUT SOURCE!"
    import sys
    sys.exit()

    
# ----- FedReadoutMode -----


if Mode == str("ZS") :

    process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

    process.MessageLogger.destinations = cms.untracked.vstring(
        "cerr",
        "DigiToRawToDigiZS",
        "info",
        "warning",
        "error"
        )
    
    process.output.fileName = "DigiToRawToDigiZS.root"

    process.DigiSource.FedRawDataMode = False
    process.DigiSource.UseFedKey = False
    
    process.SiStripDigiToRaw.FedReadoutMode = 'ZERO_SUPPRESSED'

    process.siStripDigis.UseFedKey = False
    process.newSiStripDigis.UseFedKey = False

    process.DigiValidator.TagCollection2 = "siStripDigis:ZeroSuppressed"
    process.DigiValidator.RawCollection1 = False
    process.DigiValidator.RawCollection2 = False
    
    process.newDigiValidator.TagCollection2 = "newSiStripDigis:ZeroSuppressed"
    process.newDigiValidator.RawCollection1 = False
    process.newDigiValidator.RawCollection2 = False
    
    process.testDigiValidator.TagCollection1 = "siStripDigis:ZeroSuppressed"
    process.testDigiValidator.TagCollection2 = "newSiStripDigis:ZeroSuppressed"
    process.testDigiValidator.RawCollection1 = False
    process.testDigiValidator.RawCollection2 = False
    
elif Mode == str("VR") :

    process.MessageLogger.destinations = cms.untracked.vstring(
        "cerr",
        "DigiToRawToDigiVR",
        "info",
        "warning",
        "error"
        )
    
    process.output.fileName = "DigiToRawToDigiVR.root"

    process.DigiSource.FedRawDataMode = True
    process.DigiSource.UseFedKey = False
    
    process.SiStripDigiToRaw.FedReadoutMode = 'VIRGIN_RAW'

    process.siStripDigis.UseFedKey = False
    process.newSiStripDigis.UseFedKey = False

    process.DigiValidator.RawCollection1 = False
    process.DigiValidator.TagCollection2 = "siStripDigis:VirginRaw"
    process.DigiValidator.RawCollection2 = True
    
    process.newDigiValidator.RawCollection1 = False
    process.newDigiValidator.TagCollection2 = "newSiStripDigis:VirginRaw"
    process.newDigiValidator.RawCollection2 = True
    
    process.testDigiValidator.TagCollection1 = "siStripDigis:VirginRaw"
    process.testDigiValidator.RawCollection1 = True
    process.testDigiValidator.TagCollection2 = "newSiStripDigis:VirginRaw"
    process.testDigiValidator.RawCollection2 = True
    
elif Mode == str("PR") :

    process.MessageLogger.destinations = cms.untracked.vstring(
        "cerr",
        "DigiToRawToDigiPR",
        "info",
        "warning",
        "error"
        )
    
    process.output.fileName = "DigiToRawToDigiPR.root"

    process.DigiSource.FedRawDataMode = True
    process.DigiSource.UseFedKey = False
    
    process.SiStripDigiToRaw.FedReadoutMode = 'PROCESSED_RAW'

    process.siStripDigis.UseFedKey = False
    process.newSiStripDigis.UseFedKey = False

    process.DigiValidator.RawCollection1 = False
    process.DigiValidator.TagCollection2 = "siStripDigis:ProcessedRaw"
    process.DigiValidator.RawCollection2 = True
    
    process.newDigiValidator.RawCollection1 = False
    process.newDigiValidator.TagCollection2 = "newSiStripDigis:ProcessedRaw"
    process.newDigiValidator.RawCollection2 = True
    
    process.testDigiValidator.TagCollection1 = "siStripDigis:ProcessedRaw"
    process.testDigiValidator.RawCollection1 = True
    process.testDigiValidator.TagCollection2 = "newSiStripDigis:ProcessedRaw"
    process.testDigiValidator.RawCollection2 = True
    
elif Mode == str("FK") :

    process.MessageLogger.destinations = cms.untracked.vstring(
        "cerr",
        "DigiToRawToDigiFK",
        "info",
        "warning",
        "error"
        )
    
    process.output.fileName = "DigiToRawToDigiFK.root"
    
    process.DigiSource.FedRawDataMode = True
    process.DigiSource.UseFedKey = True
    
    process.SiStripDigiToRaw.FedReadoutMode = 'VIRGIN_RAW'
    process.SiStripDigiToRaw.UseFedKey = True

    process.siStripDigis.UseFedKey = True
    process.newSiStripDigis.UseFedKey = True
    
    process.DigiValidator.RawCollection1 = False
    process.DigiValidator.TagCollection2 = "siStripDigis:VirginRaw"
    process.DigiValidator.RawCollection2 = True
    
    process.newDigiValidator.RawCollection1 = False
    process.newDigiValidator.TagCollection2 = "newSiStripDigis:VirginRaw"
    process.newDigiValidator.RawCollection2 = True
    
    process.testDigiValidator.TagCollection1 = "siStripDigis:VirginRaw"
    process.testDigiValidator.RawCollection1 = True
    process.testDigiValidator.TagCollection2 = "newSiStripDigis:VirginRaw"
    process.testDigiValidator.RawCollection2 = True
    
else :

    print "UNKNOWN FED READOUT MODE!"
    import sys
    sys.exit()

    
