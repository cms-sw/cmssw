import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import os 

process = cms.Process("TEST")

process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')

#global tags for conditions data: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'MC_38Y_V8::All'

##################################################################################

# setup 'standard'  options
options = VarParsing.VarParsing ('standard')

# setup any defaults you want
options.output = 'test_out.root'
options.files = [
'/store/relval/CMSSW_3_8_1/RelValPyquen_DiJet_pt80to120_2760GeV/GEN-SIM-RECO/MC_38Y_V8-v1/0013/DE5989EA-C9A3-DF11-8ABB-001A92811738.root',
'/store/relval/CMSSW_3_8_1/RelValPyquen_DiJet_pt80to120_2760GeV/GEN-SIM-RECO/MC_38Y_V8-v1/0013/2ED099B1-C2A3-DF11-A22A-001A92971B82.root']
options.maxEvents = 1 

# get and parse the command line arguments
options.parseArguments()


##################################################################################
# Some Services

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.debugModules = ['*']  
process.MessageLogger.categories = ['HeavyIonVertexing','heavyIonHLTVertexing','MinBiasTracking']
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
process.load("RecoHI.HiTracking.secondStep_cff")                            # pair-seeded step

##############################################################################
# Output EDM File
process.load("Configuration.EventContent.EventContentHeavyIons_cff")        #load keep/drop output commands
process.output = cms.OutputModule("PoolOutputModule",
                                  process.FEVTDEBUGEventContent,
                                  fileName = cms.untracked.string(options.output)
                                  )
process.output.outputCommands.extend(["keep *_*_*_TEST"])

##################################################################################
# Sub-sequences
process.rechits = cms.Sequence(process.siPixelRecHits*process.siStripMatchedRecHits)
process.vtxreco = cms.Sequence(process.offlineBeamSpot * process.trackerlocalreco * process.hiPixelVertices)
process.pxlreco = cms.Sequence(process.vtxreco * process.hiPixel3PrimTracks)
process.trkreco = cms.Sequence(process.offlineBeamSpot * process.trackerlocalreco * process.heavyIonTracking)

# Paths
#process.reco = cms.Path(process.RawToDigi * process.trkreco)           # for RECO from RAW
process.rereco = cms.Path(process.rechits * process.heavyIonTracking    # for reRECO from RECO
                          * process.secondStep                          # plus pair-seeded iteration
                          )
process.save = cms.EndPath(process.output)

