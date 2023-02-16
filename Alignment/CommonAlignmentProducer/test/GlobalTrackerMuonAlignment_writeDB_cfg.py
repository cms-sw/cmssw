import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

def getMat(t):
  hess = []
  grad = []
  hessF = False
  gradF = False
  for i in t:
    if i.startswith("RAW Hess start"):
      hessF = True
      continue
    if i.startswith("RAW Hess end"):
      hessF = False
      continue
    if i.startswith("RAW Grad start"):
      gradF = True
      continue
    if i.startswith("RAW Grad end"):
      gradF = False
      continue
    if hessF: hess.append(np.double(i))
    if gradF: grad.append(np.double(i))
  if len(grad) == 4: hess = np.reshape(hess, (4,4))
  else: hess = np.reshape(hess, (6,6))
  return hess, grad

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
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))


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
import os
import numpy as np
fl = [x for x in os.listdir(".") if x.startswith(outName+"0") and  x.endswith(".txt")]
if options.DOF == "4":
  Hess = np.zeros(16).reshape(4,4)
  Grad = np.zeros(4)
if options.DOF == "6":
  Hess = np.zeros(36).reshape(6,6)
  Grad = np.zeros(6)
for f in fl:
  tmp = getMat(open(f))
  Hess += tmp[0]
  Grad += tmp[1]
outPar = -np.linalg.solve(Hess, -Grad)

process.load("Alignment.CommonAlignmentProducer.GlobalTrackerMuonAlignment_cfi")
process.GlobalTrackerMuonAlignment.par1 = cms.double(outPar[0])
process.GlobalTrackerMuonAlignment.par2 = cms.double(outPar[1])
process.GlobalTrackerMuonAlignment.par3 = cms.double(outPar[2])
if options.DOF == "4":
  process.GlobalTrackerMuonAlignment.par6 = cms.double(outPar[3])
elif options.DOF == "6":
  process.GlobalTrackerMuonAlignment.par4 = cms.double(outPar[3])
  process.GlobalTrackerMuonAlignment.par5 = cms.double(outPar[4])
  process.GlobalTrackerMuonAlignment.par6 = cms.double(outPar[5])
process.GlobalTrackerMuonAlignment.extPar = cms.bool(True)
process.GlobalTrackerMuonAlignment.rootOutFile = cms.string(outName+'.root')
process.GlobalTrackerMuonAlignment.txtOutFile = cms.string(outName+'.txt')

process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagator_cfi")

process.p = cms.Path(process.MuonAlignmentFromReferenceGlobalMuonRefit*process.MuonAlignmentFromReferenceGlobalMuonRefit2 + process.GlobalTrackerMuonAlignment)


### write global Rcd to DB
from CondCore.DBCommon.CondDBSetup_cfi import CondDBSetup


if len(preDB) > 3:
  process.globalPosition = cms.ESSource("PoolDBESSource", CondDBSetup,
                                       connect = cms.string('sqlite_file:'+preDB),
                                       toGet   = cms.VPSet(cms.PSet(record = cms.string("GlobalPositionRcd"), tag = cms.string("GlobalPositionRcd")))
                                       )
  
  process.es_prefer_globalPosition = cms.ESPrefer("PoolDBESSource","globalPosition")

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    CondDBSetup,
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:'+newDB),
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

