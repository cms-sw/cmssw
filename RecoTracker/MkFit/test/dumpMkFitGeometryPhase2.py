import FWCore.ParameterSet.Config as cms

### from Configuration.Eras.Era_Run3_cff import Run3
### from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
### from Configuration.ProcessModifiers.trackingMkFit_cff import trackingMkFit

### process = cms.Process('DUMP',Run3,trackingMkFit)
process = cms.Process('DUMP')

# import of standard configurations
###process.load('Configuration.StandardSequences.Services_cff')
###process.load('FWCore.MessageService.MessageLogger_cfi')
###process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
###process.load('Configuration.StandardSequences.MagneticField_cff')
###process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['run2_mc']
process.load('Configuration.Geometry.GeometryExtended2026D88Reco_cff')

##from Configuration.AlCa.GlobalTag import GlobalTag
### process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2021_realistic', '')
###process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T21', '')

###process.load('Configuration.Geometry.GeometryExtended2026D88Reco_cff')

# process.es_prefer_ZDC  = cms.ESPrefer("ZdcGeometryFromDBEP", "")
# process.es_prefer_ecb  = cms.ESPrefer("EcalBarrelGeometryFromDBEP", "")
# process.es_prefer_ece  = cms.ESPrefer("EcalEndcapGeometryFromDBEP", "")
# process.es_prefer_Hcal = cms.ESPrefer("HcalGeometryFromDBEP", "")
# process.es_prefer_calt = cms.ESPrefer("CaloTowerGeometryFromDBEP", "")
# process.es_prefer_Cstr = cms.ESPrefer("CastorGeometryFromDBEP", "")
# process.es_prefer_GEM  = cms.ESPrefer("GEMGeometryESModule", "gemGeometry")
# process.es_prefer_trk  = cms.ESPrefer("TrackerGeometricDetESModule", "trackerNumberingGeometryDB")
# process.es_prefer_trkd = cms.ESPrefer("TrackerDigiGeometryESModule", "trackerGeometryDB")
# process.es_prefer_mkf  = cms.ESPrefer("MkFitGeometryESProducer", "mkFitGeometryESProducer")

process.MessageLogger.cerr.threshold = "INFO"
process.MessageLogger.cerr.MkFitGeometryESProducer = dict(limit=-1)

process.source = cms.Source("EmptySource")
process.maxEvents.input = 1


process.add_(cms.ESProducer("MkFitGeometryESProducer"))

defaultOutputFileName="phase2-trackerinfo.bin"

# level: 0 - no printout; 1 - print layers, 2 - print modules
# outputFileName: binary dump file; no dump if empty string
process.dump = cms.EDAnalyzer("DumpMkFitGeometry",
                              level   = cms.untracked.int32(1),
                       outputFileName = cms.untracked.string(defaultOutputFileName)
                              )

print("Requesting MkFit geometry dump into file:", defaultOutputFileName, "\n");
process.p = cms.Path(process.dump)
