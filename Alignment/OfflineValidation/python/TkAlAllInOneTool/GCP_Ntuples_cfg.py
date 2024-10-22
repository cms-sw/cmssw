import FWCore.ParameterSet.Config as cms
#import FWCore.PythonUtilities.LumiList as LumiList

from FWCore.ParameterSet.VarParsing import VarParsing

import json
import os

##Define process
process = cms.Process("ValidationIntoNTuples")

##Argument parsing
options = VarParsing()
options.register("config", "", VarParsing.multiplicity.singleton, VarParsing.varType.string , "AllInOne config")

options.parseArguments()

###Set validation mode
#valiMode = "StandAlone"

##Read in AllInOne config in JSON format
if options.config == "":
    config = {"validation": {},
              "alignments": {}}
else:
    with open(options.config, "r") as configFile:
        config = json.load(configFile)

#Global tag
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag,config["alignments"]["globaltag"])

process.load("Configuration.Geometry.GeometryRecoDB_cff")

process.load("CondCore.CondDB.CondDB_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    #destinations = cms.untracked.vstring('detailedInfo', 'cout')
    destinations = cms.untracked.vstring('warnings'),
    warnings = cms.untracked.PSet(
                       threshold  = cms.untracked.string('WARNING') 
        )
) 

##Load conditions if needed
if "conditions" in config["alignments"]:
    from CalibTracker.Configuration.Common.PoolDBESSource_cfi import poolDBESSource

    for condition in config["alignments"]["conditions"]:
        setattr(
            process, 
            "conditionsIn{}".format(condition), 
            poolDBESSource.clone(
                connect = cms.string(str(config["alignments"]["conditions"][condition]["connect"])),
                toGet = cms.VPSet(
                    cms.PSet(
                        record = cms.string(str(condition)),
                        tag = cms.string(str(config["alignments"]["conditions"][condition]["tag"]))
                    )
                )
            )
        )

        setattr(process, "prefer_conditionsIn{}".format(condition), cms.ESPrefer("PoolDBESSource", "conditionsIn{}".format(condition)))


process.source = cms.Source("EmptySource",
    firstRun=cms.untracked.uint32(config["validation"]["IOV"])
    )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
print('Output file: '+config["output"]+'/Ntuples.root')

from Alignment.OfflineValidation.trackerGeometryIntoNtuples_cfi import trackerGeometryIntoNtuples as _trackerGeometryIntoNtuples

process.dump = _trackerGeometryIntoNtuples.clone(
    outputFile = str(config["output"]+'/Ntuples.root'),
    outputTreename = 'alignTree'
)

process.p = cms.Path(process.dump)  


