import FWCore.ParameterSet.Config as cms

# The VR, PR and FK modes can only be used with the TRIV source
Source = str("SIM") # Options: "TRIV", "SIM"
Mode = str("ZS")    # Options: "ZS", "VR", "PR", "FK"
Write = bool(False) # Write output to disk

if Source == str("SIM") and Mode != str("ZS") :
    print "The VR, PR and FK modes can only be used with the TRIV source!"
    import sys
    sys.exit()

process = cms.Process("DigiToRawToDigi")

process.load("EventFilter.SiStripRawToDigi.test.Validate_DigiToRawToDigi_cff")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

# ----- WriteToDisk -----


if Write == bool(True) :
    process.e = cms.EndPath( process.output )
else :
    print "Event content not written to disk!"


# ----- InputSource -----


if Source == str("TRIV") :

    process.source = cms.Source(
        "EmptySource",
        firstRun = cms.untracked.uint32(999999)
        )

    process.dummySiStripDigiToRaw.InputModuleLabel = 'DigiSource'
    process.dummySiStripDigiToRaw.InputDigiLabel = ''

    process.oldSiStripDigiToRaw.InputModuleLabel = 'DigiSource'
    process.oldSiStripDigiToRaw.InputDigiLabel = ''

    process.newSiStripDigiToRaw.InputModuleLabel = 'DigiSource'
    process.newSiStripDigiToRaw.InputDigiLabel = ''
    
    process.DigiValidator.TagCollection1 = "DigiSource"
    process.oldDigiValidator.TagCollection1 = "DigiSource"
    
    process.p = cms.Path( process.DigiSource * process.s )

elif Source == str("SIM") :

    process.source = cms.Source(
        "PoolSource",
        fileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_3_1_1/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/ECAD7ED7-966B-DE11-B4FE-000423D99CEE.root'
        )
        )
    
    process.dummySiStripDigiToRaw.InputModuleLabel = 'simSiStripDigis'
    process.dummySiStripDigiToRaw.InputDigiLabel = 'ZeroSuppressed'

    process.oldSiStripDigiToRaw.InputModuleLabel = 'simSiStripDigis'
    process.oldSiStripDigiToRaw.InputDigiLabel = 'ZeroSuppressed'

    process.newSiStripDigiToRaw.InputModuleLabel = 'simSiStripDigis'
    process.newSiStripDigiToRaw.InputDigiLabel = 'ZeroSuppressed'

    process.DigiValidator.TagCollection1 = "simSiStripDigis:ZeroSuppressed"
    process.oldDigiValidator.TagCollection1 = "simSiStripDigis:ZeroSuppressed"

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
    
    process.dummySiStripDigiToRaw.FedReadoutMode = 'ZERO_SUPPRESSED'
    process.oldSiStripDigiToRaw.FedReadoutMode = 'ZERO_SUPPRESSED'
    process.newSiStripDigiToRaw.FedReadoutMode = 'ZERO_SUPPRESSED'

    process.oldSiStripDigis.UseFedKey = False
    process.siStripDigis.UseFedKey = False

    process.oldDigiValidator.TagCollection2 = "oldSiStripDigis:ZeroSuppressed"
    process.oldDigiValidator.RawCollection1 = False
    process.oldDigiValidator.RawCollection2 = False

    process.DigiValidator.TagCollection2 = "siStripDigis:ZeroSuppressed"
    process.DigiValidator.RawCollection1 = False
    process.DigiValidator.RawCollection2 = False
    
    process.testDigiValidator.TagCollection1 = "oldSiStripDigis:ZeroSuppressed"
    process.testDigiValidator.TagCollection2 = "siStripDigis:ZeroSuppressed"
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
    
    process.dummySiStripDigiToRaw.FedReadoutMode = 'VIRGIN_RAW'
    process.oldSiStripDigiToRaw.FedReadoutMode = 'VIRGIN_RAW'
    process.newSiStripDigiToRaw.FedReadoutMode = 'VIRGIN_RAW'

    process.oldSiStripDigis.UseFedKey = False
    process.siStripDigis.UseFedKey = False
    
    process.oldDigiValidator.RawCollection1 = False
    process.oldDigiValidator.TagCollection2 = "oldSiStripDigis:VirginRaw"
    process.oldDigiValidator.RawCollection2 = True

    process.DigiValidator.RawCollection1 = False
    process.DigiValidator.TagCollection2 = "siStripDigis:VirginRaw"
    process.DigiValidator.RawCollection2 = True
    
    process.testDigiValidator.TagCollection1 = "oldSiStripDigis:VirginRaw"
    process.testDigiValidator.RawCollection1 = True
    process.testDigiValidator.TagCollection2 = "siStripDigis:VirginRaw"
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
    
    process.dummySiStripDigiToRaw.FedReadoutMode = 'PROCESSED_RAW'
    process.oldSiStripDigiToRaw.FedReadoutMode = 'PROCESSED_RAW'
    process.newSiStripDigiToRaw.FedReadoutMode = 'PROCESSED_RAW'

    process.oldSiStripDigis.UseFedKey = False
    process.siStripDigis.UseFedKey = False
    
    process.oldDigiValidator.RawCollection1 = False
    process.oldDigiValidator.TagCollection2 = "oldSiStripDigis:ProcessedRaw"
    process.oldDigiValidator.RawCollection2 = True

    process.DigiValidator.RawCollection1 = False
    process.DigiValidator.TagCollection2 = "siStripDigis:ProcessedRaw"
    process.DigiValidator.RawCollection2 = True
    
    process.testDigiValidator.TagCollection1 = "oldSiStripDigis:ProcessedRaw"
    process.testDigiValidator.RawCollection1 = True
    process.testDigiValidator.TagCollection2 = "siStripDigis:ProcessedRaw"
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
    
    process.dummySiStripDigiToRaw.FedReadoutMode = 'VIRGIN_RAW'
    process.dummySiStripDigiToRaw.UseFedKey = True
    process.oldSiStripDigiToRaw.FedReadoutMode = 'VIRGIN_RAW'
    process.oldSiStripDigiToRaw.UseFedKey = True
    process.newSiStripDigiToRaw.FedReadoutMode = 'VIRGIN_RAW'
    process.newSiStripDigiToRaw.UseFedKey = True

    process.oldSiStripDigis.UseFedKey = True
    process.siStripDigis.UseFedKey = True
    
    process.oldDigiValidator.RawCollection1 = False
    process.oldDigiValidator.TagCollection2 = "oldSiStripDigis:VirginRaw"
    process.oldDigiValidator.RawCollection2 = True

    process.DigiValidator.RawCollection1 = False
    process.DigiValidator.TagCollection2 = "siStripDigis:VirginRaw"
    process.DigiValidator.RawCollection2 = True
    
    process.testDigiValidator.TagCollection1 = "oldSiStripDigis:VirginRaw"
    process.testDigiValidator.RawCollection1 = True
    process.testDigiValidator.TagCollection2 = "siStripDigis:VirginRaw"
    process.testDigiValidator.RawCollection2 = True
    
else :

    print "UNKNOWN FED READOUT MODE!"
    import sys
    sys.exit()

    
