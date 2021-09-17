from __future__ import print_function
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import copy 

process = cms.Process("Demo")

#prepare options

options = VarParsing.VarParsing("analysis")

options.register ('globalTag',
                  "auto:run2_data",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "GlobalTag")

options.register ('runNumber',
                  306054,
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.int,           # string, int, or float
                  "run number")

options.parseArguments()


##
## MessageLogger
##
process.load('FWCore.MessageService.MessageLogger_cfi')   
process.MessageLogger.cerr.enable = False
process.MessageLogger.SiStripNoisesFromDBMiscalibrator=dict()  
process.MessageLogger.cout = cms.untracked.PSet(
    enable    = cms.untracked.bool(True),
    enableStatistics = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),                       
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(1000)
                                   ),                                                      
    SiStripNoisesFromDBMiscalibrator = cms.untracked.PSet( limit = cms.untracked.int32(-1))
    )

process.load("Configuration.Geometry.GeometryRecoDB_cff") # Ideal geometry and interface 
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag,options.globalTag, '')

print("Using Global Tag:", process.GlobalTag.globaltag._value)

##
## Empty Source
##
process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(options.runNumber),
                            numberEventsInRun = cms.untracked.uint32(1),
                            )


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

##
## Example smearing configurations
##

########### Noise ##########
# TEC: new = 5.985 old = 5.684  => ratio: 1.052
# TOB: new = 6.628 old = 6.647  => ratio: 0.997
# TIB: new = 5.491 old = 5.392  => ratio: 1.018
# TID: new = 5.259 old = 5.080  => ratio: 1.035

##
## separately partition by partition
##
byPartition = cms.VPSet(
    cms.PSet(partition = cms.string("TEC"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.2),
             smearFactor = cms.double(0.04)
             ),
    cms.PSet(partition = cms.string("TOB"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(0.94),
             smearFactor = cms.double(0.03)
             ),
    cms.PSet(partition = cms.string("TIB"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.1),
             smearFactor = cms.double(0.02)
             ),
    cms.PSet(partition = cms.string("TID"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(0.22),
             smearFactor = cms.double(0.01)
             )
    )

##
## whole Strip tracker
##

wholeTracker = cms.VPSet(
    cms.PSet(partition = cms.string("Tracker"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(True),
             scaleFactor = cms.double(1.00),
             smearFactor = cms.double(0.00)
             )
    )

##
## by Layer only one partition
##

byLayerOnlyTIB = cms.VPSet(

    ################## TIB ##################

    cms.PSet(partition = cms.string("TIB_1"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(True),
             scaleFactor = cms.double(1),
             smearFactor = cms.double(0.1)
             ),
    cms.PSet(partition = cms.string("TIB_2"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(True),
             scaleFactor = cms.double(2),
             smearFactor = cms.double(0.25)
             ),
    cms.PSet(partition = cms.string("TIB_3"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(True),
             scaleFactor = cms.double(3),
             smearFactor = cms.double(0.2)
             ),
    cms.PSet(partition = cms.string("TIB_4"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(True),
             scaleFactor = cms.double(4),
             smearFactor = cms.double(0.01)
             )
    )

##
## hierarchies
##

subsets =  cms.VPSet(
    cms.PSet(partition = cms.string("Tracker"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(True),
             scaleFactor = cms.double(0.65),
             smearFactor = cms.double(0.05)
             ),
    cms.PSet(partition = cms.string("TEC"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(True),
             scaleFactor = cms.double(1.15),
             smearFactor = cms.double(0.02)
             ),
    cms.PSet(partition = cms.string("TECP"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(True),
             scaleFactor = cms.double(1.35),
             smearFactor = cms.double(0.02)
             ),
    cms.PSet(partition = cms.string("TECP_9"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(True),
             scaleFactor = cms.double(1.55),
             smearFactor = cms.double(0.02)
             )
    )

#
# just a silly example
#
autoparams=[]
listOfLayers=["TIB_1","TIB_2","TIB_3","TIB_4","TOB_1","TOB_2","TOB_3","TOB_4","TOB_5","TOB_6","TIDM_1","TIDM_2","TIDM_3","TECM_1","TECM_2","TECM_3","TECM_4","TECM_5","TECM_6","TECM_7","TECM_8","TECM_9","TIDP_1","TIDP_2","TIDP_3","TECP_1","TECP_2","TECP_3","TECP_4","TECP_5","TECP_6","TECP_7","TECP_8","TECP_9"]

for i,ll in enumerate(listOfLayers):
    autoparams.append(
        cms.PSet(
            partition = cms.string(ll),
            doScale   = cms.bool(True),
            doSmear   = cms.bool(True),
            scaleFactor = cms.double(i*0.1),
            smearFactor = cms.double((len(listOfLayers)-i)*0.01)
            )
        )

# process.demo = cms.EDAnalyzer('SiStripNoisesFromDBMiscalibrator',
#                               params = subsets, # as a cms.VPset
#                               fillDefaults = cms.bool(False),
#                               saveMaps = cms.bool(True)      
#                               )


##
## Impot the thresholds configuration
##
import CondTools.SiStrip.MiscalibrationSettings_cff as Settings
eachLayer = copy.deepcopy(Settings.byLayer)

process.load("CondTools.SiStrip.scaleAndSmearSiStripNoises_cfi")
#process.scaleAndSmearSiStripNoises.params  = eachLayer      # as a cms.VPset
#process.scaleAndSmearSiStripNoises.params  = wholeTracker   # as a cms.VPset
#process.scaleAndSmearSiStripNoises.params  = byPartition    # as a cms.VPset
#process.scaleAndSmearSiStripNoises.params  = subsets        # as a cms.VPset
#process.scaleAndSmearSiStripNoises.params  = byLayerOnlyTIB # as a cms.VPset
process.scaleAndSmearSiStripNoises.params  = autoparams
process.scaleAndSmearSiStripNoises.fillDefaults = False     # to fill uncabled DetIds with default

##
## Database output service
##
process.load("CondCore.CondDB.CondDB_cfi")

##
## Output database (in this case local sqlite file)
##
process.CondDB.connect = 'sqlite_file:modifiedNoise_'+ process.GlobalTag.globaltag._value+'_IOV_'+str(options.runNumber)+".db"
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDB,
                                          timetype = cms.untracked.string('runnumber'),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('SiStripNoisesRcd'),
                                                                     tag = cms.string('modifiedNoise')
                                                                     )
                                                            )
                                          )

process.p = cms.Path(process.scaleAndSmearSiStripNoises)
