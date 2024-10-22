import FWCore.ParameterSet.Config as cms
import sys,os
from __future__ import print_function

arguments=sys.argv
sqlitename="ctppspixgain.db"
tagname="CTPPSPixelGainCalibrations_test"
runnumber=1
outfile="output.root"
if len(arguments)<3:
    print ("using defaults")
    print ("usage: cmsRun retrieve_sqlite_PixGainCalib_cfg.py "+sqlitename+" "+ tagname+" "+str(runnumber)+" "+outfile)
else:
    sqlitename = arguments[2]
    if len(arguments)>3: tagname= arguments[3]
    if len(arguments)>4: runnumber=int(arguments[4])
    if len(arguments)>5: outfile=arguments[5]

process = cms.Process("ProcessOne")

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(runnumber),
    lastValue = cms.uint64(runnumber),
    interval = cms.uint64(1)
)

#Database output service
process.load("CondCore.CondDB.CondDB_cfi")

# input database (in this case local sqlite file)
process.CondDB.connect = 'sqlite_file:'+sqlitename


process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDB,
    DumpStat=cms.untracked.bool(True),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('CTPPSPixelGainCalibrationsRcd'),
        tag = cms.string(tagname)
    )),
)



process.myprodtest = cms.EDAnalyzer("CTPPSPixGainCalibsESAnalyzer",
                                    outputrootfile=cms.untracked.string(outfile)
                                    )

process.p = cms.Path(process.myprodtest)


