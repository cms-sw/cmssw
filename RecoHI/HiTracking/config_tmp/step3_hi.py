# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step3 --conditions auto:phase1_2017_realistic -n 10 --era Run2_2017 --eventcontent RECOSIM --runUnscheduled -s RAW2DIGI,L1Reco,RECO --datatier GEN-SIM-RECO --geometry DB:Extended --filein file:step2.root --fileout file:step3.root
import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('RECO',eras.Run2_2017)

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",ignoreTotal = cms.untracked.int32(1) )
process.Timing = cms.Service("Timing")

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContentHeavyIons_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.ReconstructionHeavyIons_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#'file:/afs/cern.ch/user/a/abaty/work/public/2017_Phase1_Tracking_Samples/Feb_20_2017/QCD_80_120_DIGIRAW.root'
#'file:/afs/cern.ch/user/a/abaty/work/public/2017_Phase1_Tracking_Samples/Feb_20_2017/Hydjet_RAW.root'
'/store/user/abaty/TrackingPhase1_MC/Hydjet_RAW/Baty_2017Phase1TrackingGeom_Hydjet_GEN/Baty_2017Phase1TrackingGeom_Hydjet_RAW/170316_162001/0000/PbPbstep2_DIGI2017_1.root'
),
    secondaryFileNames = cms.untracked.vstring()
)

#begin hack for MVAs in GT, we will change later
from CondCore.DBCommon.CondDBSetup_cfi import *
process.gbrforest = cms.ESSource("PoolDBESSource",CondDBSetup,
                                 toGet = cms.VPSet(
        cms.PSet( record = cms.string('GBRWrapperRcd'),
                  tag= cms.string('GBRForest_HIMVASelectorIter4_v0_offline'),
                  label  = cms.untracked.string('HIMVASelectorIter4')
                  ),
        cms.PSet( record = cms.string('GBRWrapperRcd'),
                  tag= cms.string('GBRForest_HIMVASelectorIter5_v0_offline'),
                  label  = cms.untracked.string('HIMVASelectorIter5')
                  ),
        cms.PSet( record = cms.string('GBRWrapperRcd'),
                  tag= cms.string('GBRForest_HIMVASelectorIter6_v0_offline'),
                  label  = cms.untracked.string('HIMVASelectorIter6')
                  ),
        cms.PSet( record = cms.string('GBRWrapperRcd'),
                  tag= cms.string('GBRForest_HIMVASelectorIter7_v0_offline'),
                  label  = cms.untracked.string('HIMVASelectorIter7')
                  ),
        cms.PSet( record = cms.string('GBRWrapperRcd'),
                  tag= cms.string('GBRForest_HIMVASelectorIter7_v0_offline'),#FIXME HIMVA for lowPtQuadStep, whatever iteration it will be called
                  label  = cms.untracked.string('HIMVASelectorIter8')
                  ),
        cms.PSet( record = cms.string('GBRWrapperRcd'),
                  tag= cms.string('GBRForest_HIMVASelectorIter7_v0_offline'),#FIXME HIMVA for highPtTripletStep, whatever iteration it will be called
                  label  = cms.untracked.string('HIMVASelectorIter9')
                  ),
	cms.PSet( record = cms.string('GBRWrapperRcd'),
                  tag= cms.string('GBRForest_HIMVASelectorIter7_v0_offline'),#FIXME HIMVA for detachedQuadStep, whatever iteration it will be called
                  label  = cms.untracked.string('HIMVASelectorIter10')
                  )
        
	),
        connect =cms.string('frontier://FrontierProd/CMS_CONDITIONS')
)
process.es_prefer_forest = cms.ESPrefer("PoolDBESSource","gbrforest")
#end hack

process.options = cms.untracked.PSet(
    allowUnscheduled = cms.untracked.bool(True)
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step3 nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition
process.RECOSIMoutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-RECO'),
        filterName = cms.untracked.string('')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    fileName = cms.untracked.string('file:step3.root'),
    outputCommands = process.RECODEBUGEventContent.outputCommands, #RECODEBUG for more info
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition
# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2017_realistic', '')
process.GlobalTag = GlobalTag(process.GlobalTag, '90X_upgrade2017_realistic_v6', '')

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstructionHeavyIons)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.RECOSIMoutput_step = cms.EndPath(process.RECOSIMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.L1Reco_step,process.reconstruction_step,process.endjob_step,process.RECOSIMoutput_step)

#do not add changes to your config after this point (unless you know what you are doing)
from FWCore.ParameterSet.Utilities import convertToUnscheduled
process=convertToUnscheduled(process)
from FWCore.ParameterSet.Utilities import cleanUnscheduled
process=cleanUnscheduled(process)


# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
