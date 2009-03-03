
import FWCore.ParameterSet.Config as cms

process = cms.Process( "laserAlignment" )

process.load( "Geometry.CMSCommonData.cmsIdealGeometryXML_cfi" )
process.load( "CondCore.DBCommon.CondDBSetup_cfi" )


## message logger
process.MessageLogger = cms.Service( "MessageLogger",
  debugModules = cms.untracked.vstring( 'LaserAlignment' ),
  cerr = cms.untracked.PSet(
    threshold = cms.untracked.string( 'ERROR' )
  ),
  cout = cms.untracked.PSet(
    threshold = cms.untracked.string( 'INFO' )
  ),
  destinations = cms.untracked.vstring( 'cout', 'cerr' )
)

## all db records
process.allSource = cms.ESSource( "PoolDBESSource",
  process.CondDBSetup,
  connect = cms.string( 'frontier://FrontierProd/CMS_COND_20X_GLOBALTAG' ),
  globaltag = cms.string( 'IDEAL_v2::All' )
)

## get the tracker alignment records from this file
process.alignmentSource = cms.ESSource( "PoolDBESSource",
  process.CondDBSetup,
  timetype = cms.string( 'runnumber' ),
  toGet = cms.VPSet(
    cms.PSet(
      record = cms.string( 'TrackerAlignmentRcd' ),
      tag = cms.string( 'Alignments' )
    ), 
    cms.PSet(
      record = cms.string( 'TrackerAlignmentErrorRcd' ),
      tag = cms.string( 'AlignmentErrors' )
    )
  ),
  connect = cms.string( 'sqlite_file:/afs/cern.ch/user/o/olzem/cms/cmssw/CMSSW_2_2_1/src/Alignment/LaserAlignment/test/Alignments_S.db' )
)

## prefer the alignment record
process.prefer( "alignmentSource" )


process.load( "Geometry.TrackerGeometryBuilder.trackerGeometry_cfi" )
process.TrackerDigiGeometryESModule.applyAlignment = True

process.load( "Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi" )


## fast standalone reco output: an sql file
process.load( "CondCore.DBCommon.CondDBCommon_cfi" )
process.CondDBCommon.connect = 'sqlite_file:Alignments.db'
process.PoolDBOutputService = cms.Service( "PoolDBOutputService",
  process.CondDBCommon,
  toPut = cms.VPSet(
    cms.PSet(
      record = cms.string( 'TrackerAlignmentRcd' ),
      tag = cms.string( 'Alignments' )
    ), 
    cms.PSet(
      record = cms.string( 'TrackerAlignmentErrorRcd' ),
      tag = cms.string( 'AlignmentErrors' )
    )
  )
)


## input files
process.source = cms.Source( "PoolSource",
  fileNames = cms.untracked.vstring(
    'file:/afs/cern.ch/user/o/olzem/scratch0/LaserEvents.SIM-DIGI.1136.root'
    #'file:/afs/cern.ch/user/o/olzem/scratch0/cms/las/prod/nt/TkAlLAS_0.root',
    #'file:/afs/cern.ch/user/o/olzem/scratch0/cms/las/prod/nt/TkAlLAS_1.root',
    #'file:/afs/cern.ch/user/o/olzem/scratch0/cms/las/prod/nt/TkAlLAS_2.root',
    #'file:/afs/cern.ch/user/o/olzem/scratch0/cms/las/prod/nt/TkAlLAS_3.root',
    #'file:/afs/cern.ch/user/o/olzem/scratch0/cms/las/prod/nt/TkAlLAS_4.root',
    #'file:/afs/cern.ch/user/o/olzem/scratch0/cms/las/prod/nt/TkAlLAS_5.root',
    #'file:/afs/cern.ch/user/o/olzem/scratch0/cms/las/prod/nt/TkAlLAS_6.root',
    #'file:/afs/cern.ch/user/o/olzem/scratch0/cms/las/prod/nt/TkAlLAS_7.root',
    #'file:/afs/cern.ch/user/o/olzem/scratch0/cms/las/prod/nt/TkAlLAS_8.root',
    #'file:/afs/cern.ch/user/o/olzem/scratch0/cms/las/prod/nt/TkAlLAS_9.root',
    #'file:/afs/cern.ch/user/o/olzem/scratch0/cms/las/prod/nt/TkAlLAS_10.root',
    #'file:/afs/cern.ch/user/o/olzem/scratch0/cms/las/prod/nt/TkAlLAS_11.root',
    #'file:/afs/cern.ch/user/o/olzem/scratch0/cms/las/prod/nt/TkAlLAS_12.root',
    #'file:/afs/cern.ch/user/o/olzem/scratch0/cms/las/prod/nt/TkAlLAS_13.root',
    #'file:/afs/cern.ch/user/o/olzem/scratch0/cms/las/prod/nt/TkAlLAS_14.root',
    #'file:/afs/cern.ch/user/o/olzem/scratch0/cms/las/prod/nt/TkAlLAS_15.root'
  )

#    '/store/mc/Summer08/TrackerLaser/ALCARECO/IDEAL_V2_TkAlLAS_v1/0053/0A045025-DC5C-DD11-BACC-001E0B477F28.root', 
#    '/store/mc/Summer08/TrackerLaser/ALCARECO/IDEAL_V2_TkAlLAS_v1/0053/0C24CE61-DC5C-DD11-A849-001E0B479F9C.root', 
#    '/store/mc/Summer08/TrackerLaser/ALCARECO/IDEAL_V2_TkAlLAS_v1/0053/0ED0A72F-DC5C-DD11-A382-001F29C49310.root', 
#    '/store/mc/Summer08/TrackerLaser/ALCARECO/IDEAL_V2_TkAlLAS_v1/0053/1615BCCE-DB5C-DD11-A1FB-001E0B48D104.root', 
#    '/store/mc/Summer08/TrackerLaser/ALCARECO/IDEAL_V2_TkAlLAS_v1/0053/261A6565-DC5C-DD11-9162-001E0B5F68AA.root', 
#    '/store/mc/Summer08/TrackerLaser/ALCARECO/IDEAL_V2_TkAlLAS_v1/0053/34328589-DB5C-DD11-893B-0018FE290052.root', 
#    '/store/mc/Summer08/TrackerLaser/ALCARECO/IDEAL_V2_TkAlLAS_v1/0053/4AC4F4D0-DB5C-DD11-A00D-001E0B5F68AA.root', 
#    '/store/mc/Summer08/TrackerLaser/ALCARECO/IDEAL_V2_TkAlLAS_v1/0053/88818224-DC5C-DD11-B625-001E0B48D104.root', 
#    '/store/mc/Summer08/TrackerLaser/ALCARECO/IDEAL_V2_TkAlLAS_v1/0053/8AF6D667-DC5C-DD11-B8C2-001F296A7696.root', 
#    '/store/mc/Summer08/TrackerLaser/ALCARECO/IDEAL_V2_TkAlLAS_v1/0053/9EA95865-DC5C-DD11-95AF-001F29C450E2.root', 
#    '/store/mc/Summer08/TrackerLaser/ALCARECO/IDEAL_V2_TkAlLAS_v1/0053/C23D6735-DC5C-DD11-BA1B-001F296BD566.root', 
#    '/store/mc/Summer08/TrackerLaser/ALCARECO/IDEAL_V2_TkAlLAS_v1/0053/FEBDBFDF-DB5C-DD11-8B68-001CC443B7B8.root', 
#    '/store/mc/Summer08/TrackerLaser/ALCARECO/IDEAL_V2_TkAlLAS_v1/0054/08225C59-DF5C-DD11-9C1C-001F29C4D344.root', 
#    '/store/mc/Summer08/TrackerLaser/ALCARECO/IDEAL_V2_TkAlLAS_v1/0054/10D231C3-DD5C-DD11-A38C-001F29C4A3A2.root', 
#    '/store/mc/Summer08/TrackerLaser/ALCARECO/IDEAL_V2_TkAlLAS_v1/0054/18D4FE5D-DF5C-DD11-8659-0018FE28BF02.root', 
#    '/store/mc/Summer08/TrackerLaser/ALCARECO/IDEAL_V2_TkAlLAS_v1/0054/34C872C7-DD5C-DD11-9021-0018FE2940F4.root', 
#    '/store/mc/Summer08/TrackerLaser/ALCARECO/IDEAL_V2_TkAlLAS_v1/0054/4A2C7553-DF5C-DD11-AE40-0018FE286D76.root', 
#    '/store/mc/Summer08/TrackerLaser/ALCARECO/IDEAL_V2_TkAlLAS_v1/0054/52E75EC6-DD5C-DD11-9050-001E0B476FA2.root', 
#    '/store/mc/Summer08/TrackerLaser/ALCARECO/IDEAL_V2_TkAlLAS_v1/0054/761362C8-DD5C-DD11-8DBA-001F29C4A30E.root', 
#    '/store/mc/Summer08/TrackerLaser/ALCARECO/IDEAL_V2_TkAlLAS_v1/0054/BE54F252-DF5C-DD11-9948-0018FE28BF24.root', 
#    '/store/mc/Summer08/TrackerLaser/ALCARECO/IDEAL_V2_TkAlLAS_v1/0054/E22E9BC6-DD5C-DD11-948C-001F296A52A4.root', 
#    '/store/mc/Summer08/TrackerLaser/ALCARECO/IDEAL_V2_TkAlLAS_v1/0054/F817D8D2-E05C-DD11-B2F2-0018FE28BEB8.root', 
#    '/store/mc/Summer08/TrackerLaser/ALCARECO/IDEAL_V2_TkAlLAS_v1/0054/F8FF6F59-DF5C-DD11-B93F-001E0B5FA53A.root', 
#    '/store/mc/Summer08/TrackerLaser/ALCARECO/IDEAL_V2_TkAlLAS_v1/0054/FCD96445-DF5C-DD11-AE4B-001E0B48610E.root' )
)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32( -1 )
)

## the LaserAlignment module
process.load( "Alignment.LaserAlignment.LaserAlignment_cfi" )
process.LaserAlignment.DigiProducersList = cms.VPSet(
  cms.PSet(
    DigiLabel = cms.string( 'ZeroSuppressed' ),
    DigiProducer = cms.string( 'simSiStripDigis' ),
    DigiType = cms.string( 'Processed' )
  )
)
process.LaserAlignment.SaveToDbase = False
process.LaserAlignment.SaveHistograms = True
process.LaserAlignment.SubtractPedestals = False
process.LaserAlignment.UpdateFromInputGeometry = False
process.LaserAlignment.EnableJudgeZeroFilter = False

## special parameters for LaserAlignment
# process.LaserAlignment.ForceFitterToNominalStrips = True


## the output file containing the TkLasBeamCollection
## for the track based interface
process.out = cms.OutputModule( "PoolOutputModule",
  fileName = cms.untracked.string( 'tkLasBeams.root' ),
  outputCommands = cms.untracked.vstring(
    'drop *',
    "keep TkLasBeams_*_*_*"
  )
)


## for debugging
process.dump = cms.EDAnalyzer("EventContentAnalyzer")

process.alignment = cms.Sequence( process.LaserAlignment )
process.laser = cms.Path( process.alignment )
process.e = cms.EndPath( process.out )









