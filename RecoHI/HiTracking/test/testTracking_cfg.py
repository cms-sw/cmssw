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
process.GlobalTag.globaltag = 'MC_31X_V5::All'

##################################################################################

# setup 'standard'  options
options = VarParsing.VarParsing ('standard')

# setup any defaults you want
options.output = 'test_out.root'
#options.files= 'dcache:/pnfs/cmsaf.mit.edu/t2bat/cms/store/mc/Summer09/Hydjet_MinBias_4TeV/GEN-SIM-RAW/MC_31X_V3-GaussianVtx_312_ver1/0005/FECC5F18-1982-DE11-ACF9-001EC94BA3AE.root'
#options.files= 'rfio:/castor/cern.ch/cms/store/relval/CMSSW_3_2_4/RelValHydjetQ_MinBias_4TeV/GEN-SIM-RAW/MC_31X_V3-v1/0010/D62F586C-4E84-DE11-80D3-000423D98E54.root'
options.files= 'rfio:/castor/cern.ch/cms/store/relval/CMSSW_3_2_4/RelValHydjetQ_B0_4TeV/GEN-SIM-RAW/MC_31X_V3-v1/0010/FC70A0E1-6A84-DE11-AE66-000423D98DB4.root'
options.maxEvents = 1 

# get and parse the command line arguments
options.parseArguments()


##################################################################################
# Some Services

process.load("FWCore.MessageService.MessageLogger_cfi")	
	   
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
process.load("Configuration.StandardSequences.RawToDigi_cff")			# RawToDigi
process.load("RecoHI.Configuration.Reconstruction_HI_cff")              # full heavy ion reconstruction

##############################################################################
# Output EDM File
process.load("Configuration.EventContent.EventContent_cff") #load keep/drop output commands
process.load("RecoHI.HiTracking.RecoHiTracker_EventContent_cff")
process.RECODEBUGEventContent.outputCommands.extend(process.RecoHiTrackerRECO.outputCommands)
process.output = cms.OutputModule("PoolOutputModule",
                                  process.RECODEBUGEventContent,
                                  compressionLevel = cms.untracked.int32(2),
                                  commitInterval = cms.untracked.uint32(1),
                                  fileName = cms.untracked.string(options.output)
                                  )

##################################################################################
# Paths
process.trkreco = cms.Sequence(process.offlineBeamSpot * process.trackerlocalreco * process.heavyIonTracking)
process.p = cms.Path(process.RawToDigi * process.trkreco)
process.save = cms.EndPath(process.output)

