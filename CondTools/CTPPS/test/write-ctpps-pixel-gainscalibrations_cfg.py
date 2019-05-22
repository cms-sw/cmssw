from __future__ import print_function
import FWCore.ParameterSet.Config as cms
import sys,os

arguments = sys.argv
infile = "Gains.root"
tagname = "CTPPSPixelGainCalibrations_test"
sqlitename = "ctppsgains_test.db"
runnumber = 1
doDummy = False
gainlo = 0.0
gainhi = 100.0
minNp = 3
if len(arguments)<3:
    print ("using default values")
    mystr = "usage: cmsRun write-ctpps-pixel-gainscalibrations_cfg.py "+ infile +" "+ tagname +" "+ sqlitename +" "+ str(runnumber) +" "+ str(doDummy) +" "+ str(gainlo) +" "+ str(gainhi) +" "+ str(minNp)
    print (mystr)
else:
    infile = arguments[2]
    if len(arguments)>3: tagname    = arguments[3]
    if len(arguments)>4: sqlitename = arguments[4]
    if len(arguments)>5:
        runnumber  = int(arguments[5])
        print ("runno = ",runnumber)
    if len(arguments)>6:
        doDummy    = (arguments[6].lower()=="true")
        print ("useDummyValues = ",doDummy)
    if len(arguments)>7:
        gainlo     = float(arguments[7])
        print ("gainLowLimit = ",gainlo)
    if len(arguments)>8:
        gainhi     = float(arguments[8])
        print ("gainHighLimit = ",gainhi)
    if len(arguments)>9:
        minNp      = int(arguments[9])
        print ("minimumNpFit = ",minNp)



process = cms.Process("CTPPSPixelGainDB")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.pixGainDB = cms.EDAnalyzer("WriteCTPPSPixGainCalibrations",
                                   inputrootfile = cms.untracked.string(infile),
                                   record = cms.untracked.string('CTPPSPixelGainCalibrationsRcd'),
                                   useDummyValues = cms.untracked.bool(doDummy),
                                   gainLowLimit = cms.untracked.double(gainlo),
                                   gainHighLimit=cms.untracked.double(gainhi),
                                   minimumNpfit = cms.untracked.int32(minNp)
                                   )


process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(runnumber),
    lastValue = cms.uint64(runnumber),
    interval = cms.uint64(1)
    )

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(10),
        authenticationPath = cms.untracked.string('.')
        ),
    toPut = cms.VPSet(
        cms.PSet(
            record = cms.string('CTPPSPixelGainCalibrationsRcd'),
            tag = cms.string(tagname)
            )
        ),
    connect = cms.string('sqlite_file:'+sqlitename)
)

process.p = cms.Path(process.pixGainDB)
