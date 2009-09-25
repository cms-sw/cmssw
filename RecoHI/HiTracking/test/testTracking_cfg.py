import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import os 

process = cms.Process("RECO")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")

#global tags for conditions data: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'MC_31X_V8::All'

##################################################################################

# setup 'standard'  options
options = VarParsing.VarParsing ('standard')

# setup any defaults you want
options.output = 'test_out.root'
options.files= '/store/relval/CMSSW_3_3_0_pre3/RelValHydjetQ_MinBias_4TeV/GEN-SIM-RAW/MC_31X_V8-v1/0015/DC571B73-43A1-DE11-BD0C-000423D98804.root'
#options.files= '/store/relval/CMSSW_3_3_0_pre3/RelValHydjetQ_B0_4TeV/GEN-SIM-RAW/MC_31X_V8-v1/0015/FE2B9E7D-4CA1-DE11-9FA1-000423D6CA02.root'
options.maxEvents = 1 

# get and parse the command line arguments
options.parseArguments()


##################################################################################
# Some Services

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.debugModules = ['*']  
process.MessageLogger.categories = ['HeavyIonVertexing','heavyIonHLTVertexing']
process.MessageLogger.cerr = cms.untracked.PSet(
    threshold = cms.untracked.string('DEBUG'),
    DEBUG = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    ),
    INFO = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    ),
    HeavyIonVertexing = cms.untracked.PSet(
        limit = cms.untracked.int32(-1)
	),
    heavyIonHLTVertexing = cms.untracked.PSet(
        limit = cms.untracked.int32(-1)
    )
)
	   
process.SimpleMemoryCheck = cms.Service('SimpleMemoryCheck',
                                        ignoreTotal=cms.untracked.int32(0),
                                        oncePerEventMode = cms.untracked.bool(False)
                                        )

process.Timing = cms.Service("Timing")

##################################################################################
# Input Source
process.source = cms.Source('PoolSource',
	fileNames = cms.untracked.vstring(options.files)
)
							
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents)
)

##################################################################################
#Reconstruction			
process.load("Configuration.StandardSequences.RawToDigi_cff")		    # RawToDigi
process.load("Configuration.StandardSequences.ReconstructionHeavyIons_cff") # full heavy ion reconstruction

#process.hiSelectedTracks.minZCut = 0.005

##############################################################################
# Output EDM File
process.load("Configuration.EventContent.EventContentHeavyIons_cff")        #load keep/drop output commands
process.output = cms.OutputModule("PoolOutputModule",
                                  process.FEVTDEBUGEventContent,
                                  compressionLevel = cms.untracked.int32(2),
                                  commitInterval = cms.untracked.uint32(1),
                                  fileName = cms.untracked.string(options.output)
                                  )

##################################################################################
# Paths
process.vtxreco = cms.Sequence(process.offlineBeamSpot * process.trackerlocalreco * process.hiPixelVertices)
process.pxlreco = cms.Sequence(process.vtxreco * process.hiPixel3PrimTracks)
process.trkreco = cms.Sequence(process.offlineBeamSpot * process.trackerlocalreco * process.heavyIonTracking)
process.p = cms.Path(process.RawToDigi * process.trkreco)
process.save = cms.EndPath(process.output)

