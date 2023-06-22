import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("writeBeamProfile2DB")

options = VarParsing.VarParsing()
options.register('unitTest',
                 False, # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.bool, # string, int, or float
                 "are we running the unit test?")
options.register('inputTag',
                 "myTagName", # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.string, # string, int, or float
                 "output tag name")
options.parseArguments()


process.load("FWCore.MessageLogger.MessageLogger_cfi")
from CondCore.CondDB.CondDB_cfi import *

if options.unitTest :
    tag_name = 'simBS_tag'
else:
    tag_name = options.inputTag

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

from CondTools.BeamSpot.beamProfile2DBWriter_cfi import beamProfile2DBWriter
process.load("IOMC.EventVertexGenerators.VtxSmearedParameters_cfi")

# Get the PSets that have BetaStar as a parameter
psets_with_BetaStar = []
psets_names = []
for psetName in process.__dict__:
    pset = getattr(process, psetName)
    if isinstance(pset, cms.PSet) and "BetaStar" in pset.parameterNames_():
        print(psetName)
        psets_names.append(psetName)
        psets_with_BetaStar.append(pset)

# Create a VPSet to store the parameter sets
myVPSet = cms.VPSet()

for i, pset in enumerate(psets_with_BetaStar):
    cloneName = 'BeamProfile2DBWriter_' + str(i+1)  # Unique clone name
    setattr(process, cloneName, beamProfile2DBWriter.clone(pset,
                                                           recordName = cms.string(psets_names[i])))

    # Create a path for each clone and add the clone to it
    pathName = 'Path_' + str(i+1)  # Unique path name
    setattr(process, pathName, cms.Path(getattr(process, cloneName)))

    myPSet = cms.PSet(
        record =  cms.string(psets_names[i]),
        tag = cms.string(psets_names[i])
    )
    
    myVPSet.append(myPSet)

#################################
# Produce a SQLITE FILE
#################################
CondDBSimBeamSpotObjects = CondDB.clone(connect = cms.string('sqlite_file:test_%s.db' % tag_name)) # choose an output name
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          CondDBSimBeamSpotObjects,
                                          timetype = cms.untracked.string('runnumber'),
                                          toPut = myVPSet,
                                          loadBlobStreamer = cms.untracked.bool(False))

# Add an end path
process.end = cms.EndPath()

process.schedule = cms.Schedule()
for i, pset in enumerate(psets_with_BetaStar):
    pathName = 'Path_' + str(i+1)  # Unique path name
    process.schedule.append(getattr(process, pathName))
process.schedule.append(process.end)


