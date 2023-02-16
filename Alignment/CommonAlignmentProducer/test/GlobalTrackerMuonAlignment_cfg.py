import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing()
options.register('start',
                 default = '0',
                 mytype = VarParsing.VarParsing.varType.string,
                 info = "start")
options.register('end',
                 default = '-1',
                 mytype = VarParsing.VarParsing.varType.string,
                 info = "end")
options.register('outputGPR',
                 default = "test.db",
                 mytype = VarParsing.VarParsing.varType.string,
                 info = "output file name")
options.register('TBMADB',
                 default = "",
                 mytype = VarParsing.VarParsing.varType.string,
                 info = "TBMADB file")
options.register('inputGPR',
                 default = "",
                 mytype = VarParsing.VarParsing.varType.string,
                 info = "input file name")
options.register('inputGT',
                 default = "",
                 mytype = VarParsing.VarParsing.varType.string,
                 info = "input GT")
options.register('fileList',
                 default = "",
                 mytype = VarParsing.VarParsing.varType.string,
                 info = "input file list")
options.register('DOF',
                 default = "4",
                 mytype = VarParsing.VarParsing.varType.string,
                 info = "degree of freedom")
options.register('Global',
                 default = False,
                 mytype = VarParsing.VarParsing.varType.string,
                 info = "use global cordi.")
options.register('Barrel',
                 default = False,
                 mytype = VarParsing.VarParsing.varType.string,
                 info = "use barrel only")
options.parseArguments()

from Configuration.Eras.Era_Run3_cff import Run3

process = cms.Process("GlobalAlignment", Run3)

process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load('TrackingTools.TransientTrack.TransientTrackBuilder_cfi')

process.MuonNumberingInitialization = cms.ESProducer("MuonNumberingInitialization")
process.MuonNumberingRecord = cms.ESSource( "EmptyESSource",
    recordName = cms.string( "MuonNumberingRecord" ),
    iovIsRunNotTime = cms.bool( True ),
    firstValid = cms.vuint32( 1 )
)

process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = options.inputGT

###  number of events ###
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))


from TrackingTools.TrackRefitter.globalMuonTrajectories_cff import *
process.MuonAlignmentFromReferenceGlobalMuonRefit = globalMuons.clone()
process.MuonAlignmentFromReferenceGlobalMuonRefit.Tracks = cms.InputTag("ALCARECOMuAlCalIsolatedMu:TrackerOnly")
process.MuonAlignmentFromReferenceGlobalMuonRefit.TrackTransformer.RefitRPCHits = cms.bool(False)

process.MuonAlignmentFromReferenceGlobalMuonRefit2 = globalMuons.clone()
process.MuonAlignmentFromReferenceGlobalMuonRefit2.Tracks = cms.InputTag("ALCARECOMuAlCalIsolatedMu:StandAlone")
process.MuonAlignmentFromReferenceGlobalMuonRefit2.TrackTransformer.RefitRPCHits = cms.bool(False)


preDB = options.inputGPR
newDB = options.outputGPR
outName = newDB.split('.')[0]
process.load("Alignment.CommonAlignmentProducer.GlobalTrackerMuonAlignment_cfi")
process.GlobalTrackerMuonAlignment.writeDB = cms.bool(False)
if options.DOF == "4":
  process.GlobalTrackerMuonAlignment.Local4 = cms.bool(True)
if options.DOF == "6":
  if options.Global: process.GlobalTrackerMuonAlignment.Global = cms.bool(True)
  else: process.GlobalTrackerMuonAlignment.Local = cms.bool(True)
  #process.GlobalTrackerMuonAlignment.mixLocalGlobal = cms.bool(True)
if options.Barrel: process.GlobalTrackerMuonAlignment.barrel = cms.bool(True)
process.GlobalTrackerMuonAlignment.rootOutFile = cms.string(outName+'.root')
process.GlobalTrackerMuonAlignment.txtOutFile = cms.string(outName+'.txt')


### debug  isolated/cosmic  refit
# isolated muon

process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagator_cfi")

#process.p = cms.Path(process.MuonAlignmentFromReferenceGlobalMuonRefit + process.GlobalTrackerMuonAlignment)
process.p = cms.Path(process.MuonAlignmentFromReferenceGlobalMuonRefit*process.MuonAlignmentFromReferenceGlobalMuonRefit2 + process.GlobalTrackerMuonAlignment)


### write global Rcd to DB
from CondCore.DBCommon.CondDBSetup_cfi import CondDBSetup


if len(preDB) > 3:
  process.globalPosition = cms.ESSource("PoolDBESSource", CondDBSetup,
                                       connect = cms.string('sqlite_file:'+preDB),
                                       toGet   = cms.VPSet(cms.PSet(record = cms.string("GlobalPositionRcd"), tag = cms.string("GlobalPositionRcd")))
                                       )
  
  process.es_prefer_globalPosition = cms.ESPrefer("PoolDBESSource","globalPosition")
if len(options.TBMADB) > 3:
  process.TBMADB = cms.ESSource("PoolDBESSource", CondDBSetup,
                                     connect = cms.string('sqlite_file:'+options.TBMADB),
                                     toGet   = cms.VPSet(
                                                         cms.PSet(record = cms.string("DTAlignmentRcd"), tag = cms.string("DTAlignmentRcd")), 
                                                         cms.PSet(record = cms.string("DTAlignmentErrorExtendedRcd"), tag = cms.string("DTAlignmentErrorExtendedRcd")),
                                                         cms.PSet(record = cms.string("CSCAlignmentRcd"), tag = cms.string("CSCAlignmentRcd")),
                                                         cms.PSet(record = cms.string("CSCAlignmentErrorExtendedRcd"), tag = cms.string("CSCAlignmentErrorExtendedRcd")))
                                     )
  process.es_prefer_TBMADB = cms.ESPrefer("PoolDBESSource","TBMADB")

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    CondDBSetup,
    ### Writing to oracle needs the following shell variable setting (in zsh):
    ### export CORAL_AUTH_PATH=/afs/cern.ch/cms/DB/conddb
    ### string connect = "oracle://cms_orcoff_int2r/CMS_COND_ALIGNMENT"
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:'+newDB),
    ### untracked uint32 authenticationMethod = 1
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('GlobalPositionRcd'),
        tag = cms.string('GlobalPositionRcd')
    ))
)

process.MessageLogger.cerr.FwkReport.reportEvery = 50000

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring())

flist = open(options.fileList)
for fpath in flist:
  process.source.fileNames.append(fpath)

process.source.fileNames = process.source.fileNames[int(options.start):int(options.end)]
"""
import FWCore.PythonUtilities.LumiList as LumiList
process.source.lumisToProcess = LumiList.LumiList(filename = '/afs/cern.ch/work/m/mkizilov/private/GPR_CMSSW_12_3_0/src/Alignment/CommonAlignmentProducer/test/MK_TestRefit/Run2016H_exp_29_06_22.json').getVLuminosityBlockRange()
process.options = cms.untracked.PSet(
 SkipEvent = cms.untracked.vstring('ProductNotFound')
)
"""
