# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step3 -n 1000 --conditions auto:run2_data -s RAW2DIGI,L1Reco,RECO --datatier RECO --customise SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1 --eventcontent RECO --magField AutoFromDBCurrent --no_exec --filein /store/data/Commissioning2015/MinimumBias/RAW/v1/000/239/754/00000/94E8E718-75DB-E411-9D0E-02163E0123CC.root --scenario pp
import FWCore.ParameterSet.Config as cms

process = cms.Process('ReRECO')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')

process.MessageLogger = cms.Service("MessageLogger",  
 cout = cms.untracked.PSet(  
  default = cms.untracked.PSet( ## kill all messages in the log  
 
   limit = cms.untracked.int32(0)  
  ),  
  FwkJob = cms.untracked.PSet( ## but FwkJob category - those unlimitted  
 
   limit = cms.untracked.int32(-1)  
  )  
 ),  
 categories = cms.untracked.vstring('FwkJob'), 
 destinations = cms.untracked.vstring('cout')  
)  
 

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-10)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'/store/data/Run2015D/SingleMuon/RECO/PromptReco-v3/000/256/676/00000/2844234C-425F-E511-90F2-02163E014719.root'
),                                
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step3 nevts:1000'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')
#process.GlobalTag = GlobalTag(process.GlobalTag, 'GR_P_V54', '')
# Path and EndPath definitions
#process.raw2digi_step = cms.Path(process.RawToDigi)
#process.L1Reco_step = cms.Path(process.L1Reco)
#process.reconstruction_step = cms.Path(process.reconstruction)
#process.endjob_step = cms.EndPath(process.endOfProcess)
#process.RECOoutput_step = cms.EndPath(process.RECOoutput)

# Schedule definition
#process.schedule = cms.Schedule(process.raw2digi_step,process.L1Reco_step,process.reconstruction_step,process.endjob_step,process.RECOoutput_step)

# customisation of the process.

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.postLS1Customs
from SLHCUpgradeSimulations.Configuration.postLS1Customs import customisePostLS1 

#call to customisation function customisePostLS1 imported from SLHCUpgradeSimulations.Configuration.postLS1Customs
process = customisePostLS1(process)

# End of customisation functions

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",  
  ignoreTotal = cms.untracked.int32(1) ## default is one  
)  

process.hofilter = cms.EDFilter("HOCalibFilter",  
hoCalibVariableCollectionTag = cms.InputTag('hoCalibProducer', 'HOCalibVariableCollection')
)

process.hoCalibc = cms.EDAnalyzer("HOCalibAnalyzer",
hoCalibVariableCollectionTag = cms.InputTag('hoCalibProducer', 'HOCalibVariableCollection'),
hoInputTag = cms.InputTag('horeco'),
# hoInputTag = cms.InputTag('reducedHcalRecHits','hbhereco'),
allsignal = cms.untracked.bool(True), 
correl = cms.untracked.bool(False),  
hoinfo = cms.untracked.bool(False),  
histFit = cms.untracked.bool(False), ## true   
cosmic = cms.untracked.bool(False),  
hbtime = cms.untracked.bool(False),  
hbinfo = cms.untracked.bool(True),  
checkmap = cms.untracked.bool(False),  
noise = cms.untracked.bool(True),  
psFileName = cms.untracked.string('hocalib_test_collision.ps'),  
txtFileName = cms.untracked.string('hocalib_test_collision.txt'),  
RootFileName = cms.untracked.string('hocalib_test_collision.root'),  
  
combined = cms.untracked.bool(True),  
get_constant = cms.untracked.bool(False), #True),  
pedSuppr = cms.untracked.bool(False), ## true  
sigma = cms.untracked.double(0.05),  
get_figure = cms.untracked.bool(False) # True)  
)  
  
process.load("Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalHO_cff")  
#process.hoCalibProducer.muons = 'cosmicMuons'
process.hoCalibProducer.muons = 'muons' 
process.hoCalibProducer.hbinfo = True  
process.hoCalibProducer.hotime = True  
process.hoCalibProducer.sigma = 0.05  
process.hoCalibProducer.hoInput = 'horeco'  
process.hoCalibProducer.hbheInput = 'hbhereco'  
process.hoCalibProducer.towerInput = 'towerMaker'  
  
process.TFileService = cms.Service("TFileService",  
 fileName = cms.string('hist_test_collision.root'),   
)   
  
process.dump = cms.EDAnalyzer("EventContentAnalyzer")   
process.oout = cms.OutputModule("PoolOutputModule",   
   outputCommands = cms.untracked.vstring('keep *_*_HOCalibVariableCollection_*'),   
    fileName = cms.untracked.string('event_test_collision.root'),
        SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p1')
    )
)  

#process.p1 = cms.Path((process.RawToDigi*process.L1Reco*process.reconstruction*process.hoCalibProducer*process.hofilter*process.hoCalibc)  
#process.p1 = cms.Path((process.RawToDigi*process.L1Reco*process.reconstructionCosmics*process.hoCalibProducer*process.hoCalibc))  
#process.p1 = cms.Path((process.RawToDigi*process.L1Reco*process.reconstruction*process.hoCalibProducer*process.hoCalibc))  
process.p1 = cms.Path((process.hoCalibProducer*process.hoCalibc))  

#process.e = cms.EndPath(process.oout)
