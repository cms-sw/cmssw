#
# rechits are not persisent anymore, so one should run one of the CPEs
# on clusters ot do the track fitting. 11/08 d.k.
#
import FWCore.ParameterSet.Config as cms

process = cms.Process("recHitsTest")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('ReadPixelRecHit'),
    destinations = cms.untracked.vstring('cout'),
#    destinations = cms.untracked.vstring("log","cout"),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    )
#    log = cms.untracked.PSet(
#        threshold = cms.untracked.string('DEBUG')
#    )
)

process.source = cms.Source("PoolSource",
#    fileNames =  cms.untracked.vstring('file:/scratch/dkotlins/digis.root')
   fileNames =  cms.untracked.vstring(
    'file:/afs/cern.ch/work/d/dkotlins/public/MC/mu/pt100_71_pre7/rechits/rechits2_postls171.root'
   )
)


# a service to use root histos
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('histo.root')
)

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
# process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")# Choose the global tag here:
# 2012
#process.GlobalTag.globaltag = 'GR_P_V40::All'
# MC 2014
process.GlobalTag.globaltag = 'MC_70_V1::All'

# read rechits
process.analysis = cms.EDAnalyzer("ReadPixelRecHit",
    Verbosity = cms.untracked.bool(True),
    src = cms.InputTag("siPixelRecHits"),
)

process.p = cms.Path(process.analysis)

# test the DB object, works
#process.load("RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff")
##process.load("RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi")
##process.load("CalibTracker.SiPixelESProducers.SiPixelFakeTemplateDBObjectESSource_cfi"
##process.load("CalibTracker.SiPixelESProducers.SiPixelFakeCPEGenericErrorParmESSource_cfi"
#process.test = cms.EDAnalyzer("CPEAccessTester",
##    PixelCPE = cms.string('PixelCPEGeneric'),
#    PixelCPE = cms.string('PixelCPETemplateReco'),
#)
#process.p = cms.Path(process.test)





