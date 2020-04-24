import FWCore.ParameterSet.Config as cms

process = cms.Process("SiPixelDBTestt")

# does not run - fake conditions do not exist 
process.load("Configuration.StandardSequences.FakeConditions_cff")

#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")# Choose the global tag here:
# 2012
#process.GlobalTag.globaltag = 'GR_P_V40::All'
# MC 2913
#process.GlobalTag.globaltag = 'MC_70_V1::All'


#useFakeSource = True
useFakeSource = False
useCPEGeneric = True
#useCPEGeneric = False

process.load("RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi")

process.load("RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi")
if useCPEGeneric:
    process.siPixelRecHits.CPE = cms.string('PixelCPEGeneric')
else:
    process.siPixelRecHits.CPE = cms.string('PixelCPETemplateReco')

process.load("RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")

if useFakeSource:
    process.load("CalibTracker.SiPixelESProducers.SiPixelFakeTemplateDBObjectESSource_cfi")
    if useCPEGeneric:
        process.load("CalibTracker.SiPixelESProducers.SiPixelFakeCPEGenericErrorParmESSource_cfi")
else:
    if useCPEGeneric:
        process.load("CalibTracker.SiPixelESProducers.SiPixelFakeTemplateDBObjectESSource_cfi")
        process.PoolDBESSource = cms.ESSource("PoolDBESSource",
                                              process.CondDBSetup,
                                              loadAll = cms.bool(True),
                                              toGet = cms.VPSet(cms.PSet(
            record = cms.string('SiPixelCPEGenericErrorParmRcd'),
            tag = cms.string('SiPixelCPEGenericErrorParm')
            )),
            DBParameters = cms.PSet(messageLevel = cms.untracked.int32(0),
                                    authenticationPath = cms.untracked.string('.')),
            catalog = cms.untracked.string('file:PoolFileCatalog.xml'),
            timetype = cms.string('runnumber'),
            connect = cms.string('sqlite_file:CondTools/SiPixel/test/siPixelCPEGenericErrorParm.db'))
    else:
        process.PoolDBESSource = cms.ESSource("PoolDBESSource",
                                              process.CondDBSetup,loadAll = cms.bool(True),
                                              toGet = cms.VPSet(cms.PSet(
            record = cms.string('SiPixelTemplateDBObjectRcd'),
            tag = cms.string('SiPixelTemplateDBObject')
            )),
            DBParameters = cms.PSet(messageLevel = cms.untracked.int32(0),
                                    authenticationPath = cms.untracked.string('.')),
            catalog = cms.untracked.string('file:PoolFileCatalog.xml'),
            timetype = cms.string('runnumber'),
            connect = cms.string('sqlite_file:CondTools/SiPixel/test/siPixelTemplates.db'))
           
 
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#    '/store/relval/CMSSW_2_1_9/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/2A00EECC-A185-DD11-93A9-000423D9517C.root'
    'file:/afs/cern.ch/work/d/dkotlins/public/MC/mu/pt100/rechits/rechits1.root'

)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.MessageLogger = cms.Service("MessageLogger")
#,
 #           threshold = cms.untracked.string('ERROR')
  #  )

process.Reco = cms.Sequence(process.siPixelRecHits)
#process.Reco = cms.Sequence(process.siPixelClusters*process.siPixelRecHits)
#process.Reco = cms.Sequence(process.SiPixelTemplateDBFakeSourceReader)
process.p = cms.Path(process.Reco)

