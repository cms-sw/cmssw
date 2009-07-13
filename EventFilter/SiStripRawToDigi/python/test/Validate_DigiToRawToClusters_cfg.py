import FWCore.ParameterSet.Config as cms

Source = str("SIM") # Options: "TRIV", "SIM"
Write = bool(False) # Write output to disk

process = cms.Process("DigiToRawToClusters")

process.load("EventFilter.SiStripRawToDigi.test.Validate_DigiToRawToClusters_cff")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

process.MessageLogger.destinations = cms.untracked.vstring(
    "cerr",
    "DigiToRawToClusters",
    "info",
    "warning",
    "error"
    )

process.output.fileName = "DigiToRawToClusters.root"

if Write == bool(True) :
    process.e = cms.EndPath( process.output )
else :
    print "Event content not written to disk!"

if Source == str("TRIV") :

    process.source = cms.Source(
        "EmptySource",
        firstRun = cms.untracked.uint32(999999)
        )

    process.dummySiStripDigiToRaw.FedReadoutMode = 'ZERO_SUPPRESSED'
    process.dummySiStripDigiToRaw.InputModuleLabel = 'DigiSource'
    process.dummySiStripDigiToRaw.InputDigiLabel = ''

    process.SiStripDigiToRaw.FedReadoutMode = 'ZERO_SUPPRESSED'
    process.SiStripDigiToRaw.InputModuleLabel = 'DigiSource'
    process.SiStripDigiToRaw.InputDigiLabel = ''

    process.oldSiStripDigiToRaw.FedReadoutMode = 'ZERO_SUPPRESSED'
    process.oldSiStripDigiToRaw.InputModuleLabel = 'DigiSource'
    process.oldSiStripDigiToRaw.InputDigiLabel = ''

    process.newSiStripDigiToRaw.FedReadoutMode = 'ZERO_SUPPRESSED'
    process.newSiStripDigiToRaw.InputModuleLabel = 'DigiSource'
    process.newSiStripDigiToRaw.InputDigiLabel = ''

    process.p = cms.Path( process.DigiSource * process.s )

elif Source == str("SIM") :

    process.source = cms.Source(
        "PoolSource",
        fileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_3_1_1/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/ECAD7ED7-966B-DE11-B4FE-000423D99CEE.root'
        )
        )

    process.dummySiStripDigiToRaw.FedReadoutMode = 'ZERO_SUPPRESSED'
    process.dummySiStripDigiToRaw.InputModuleLabel = 'simSiStripDigis'
    process.dummySiStripDigiToRaw.InputDigiLabel = 'ZeroSuppressed'

    process.SiStripDigiToRaw.FedReadoutMode = 'ZERO_SUPPRESSED'
    process.SiStripDigiToRaw.InputModuleLabel = 'simSiStripDigis'
    process.SiStripDigiToRaw.InputDigiLabel = 'ZeroSuppressed'

    process.oldSiStripDigiToRaw.FedReadoutMode = 'ZERO_SUPPRESSED'
    process.oldSiStripDigiToRaw.InputModuleLabel = 'simSiStripDigis'
    process.oldSiStripDigiToRaw.InputDigiLabel = 'ZeroSuppressed'

    process.newSiStripDigiToRaw.FedReadoutMode = 'ZERO_SUPPRESSED'
    process.newSiStripDigiToRaw.InputModuleLabel = 'simSiStripDigis'
    process.newSiStripDigiToRaw.InputDigiLabel = 'ZeroSuppressed'

    process.p = cms.Path( process.s )

else :

    print "UNKNOWN INPUT SOURCE!"
    
    
