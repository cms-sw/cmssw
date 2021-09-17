# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: miniAOD-prod -s PAT --eventcontent MINIAODSIM --runUnscheduled --mc --conditions 80X_mcRun2_asymptotic_2016_TrancheIV_v4_Tr4GT_v4 --era Run2_2016 --no_exec --filein /store/relval/CMSSW_8_0_20/RelValTTbar_13/MINIAODSIM/PU25ns_80X_mcRun2_asymptotic_2016_TrancheIV_v4_Tr4GT_v4-v1/00000/A8C282AE-D37A-E611-8603-0CC47A4C8ECE.root
import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('PAT2',eras.Run2_2016)

### Example how to check timing
#process.Timing = cms.Service("Timing",
#  summaryOnly = cms.untracked.bool(False),
#  useJobReport = cms.untracked.bool(True)
#)

### Example how to check memory
#process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",ignoreTotal = cms.untracked.int32(1) )

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
#process.load('PhysicsTools.PatAlgos.slimming.metFilterPaths_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_11_0_0_pre6/RelValTTbar_13/MINIAODSIM/PU25ns_110X_upgrade2018_realistic_v3-v1/20000/F38B9A9F-4B4B-3D4B-8C9D-9A9B945194EF.root',
    ),
    secondaryFileNames = cms.untracked.vstring(),
    skipEvents = cms.untracked.uint32(145)
)

process.options = cms.untracked.PSet(
    allowUnscheduled = cms.untracked.bool(True)
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('miniAOD-prod nevts:1'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.MINIAODSIMoutput = cms.OutputModule("PoolOutputModule",
    compressionAlgorithm = cms.untracked.string('LZMA'),
    compressionLevel = cms.untracked.int32(4),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string(''),
        filterName = cms.untracked.string('')
    ),
    dropMetaData = cms.untracked.string('ALL'),
    eventAutoFlushCompressedSize = cms.untracked.int32(15728640),
    fastCloning = cms.untracked.bool(False),
    fileName = cms.untracked.string('NewMiniAOD.root'),
    outputCommands = cms.untracked.vstring('keep *'),
    overrideInputFileSplitLevels = cms.untracked.bool(True)
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
### Pick a global tag that includes the desired JER-SFs
process.GlobalTag = GlobalTag(process.GlobalTag, '102X_mcRun2_asymptotic_v7', '')

# Path and EndPath definitions
process.MINIAODSIMoutput_step = cms.EndPath(process.MINIAODSIMoutput)

process.load('Configuration.StandardSequences.Services_cff')
process.load("JetMETCorrections.Modules.JetResolutionESProducer_cfi")

### Example how to read the JER-SF from a db file
#from CondCore.DBCommon.CondDBSetup_cfi import *
#process.jer = cms.ESSource("PoolDBESSource",
#        CondDBSetup,
#        toGet = cms.VPSet(
#            # Resolution
#            cms.PSet(
#                record = cms.string('JetResolutionRcd'),
#                tag    = cms.string('JR_Autumn18_V7_MC_PtResolution_AK4PFchs'),
#                label  = cms.untracked.string('AK4PFchs_pt')
#                ),
#
#            # Scale factors
#            cms.PSet(
#                record = cms.string('JetResolutionScaleFactorRcd'),
#                tag    = cms.string('JR_Autumn18_V7_MC_SF_AK4PFchs'),
#                label  = cms.untracked.string('AK4PFchs')
#                ),
#            ),
#        connect = cms.string('sqlite:Autumn18_V7_MC.db')
#        )
#
#process.es_prefer_jer = cms.ESPrefer('PoolDBESSource', 'jer')

# example for smearing the jet resolution
process.slimmedJetsSmeared = cms.EDProducer('SmearedPATJetProducer',
       src = cms.InputTag('slimmedJets'),
       enabled = cms.bool(True),
       rho = cms.InputTag("fixedGridRhoFastjetAll"),
       algo = cms.string('AK4PFchs'),
       algopt = cms.string('AK4PFchs_pt'),
       #resolutionFile = cms.FileInPath('Autumn18_V7_MC_PtResolution_AK4PFchs.txt'),
       #scaleFactorFile = cms.FileInPath('combined_SFs_uncertSources.txt'),

       genJets = cms.InputTag('slimmedGenJets'),
       dRMax = cms.double(0.2),
       dPtMaxFactor = cms.double(3),

       debug = cms.untracked.bool(False),
   # Systematic variation
   # 0: Nominal
   # -1: -1 sigma (down variation)
   # 1: +1 sigma (up variation)
   variation = cms.int32(0),  # If not specified, default to 0
   uncertaintySource = cms.string(""), # If not specified, default to Total
       )
process.p=cms.Path(process.slimmedJetsSmeared)

# example for computing total uncertainties on jet resolution
process.slimmedJetsSmearedDown=process.slimmedJetsSmeared.clone(variation=cms.int32(-1))
process.p+=process.slimmedJetsSmearedDown
process.slimmedJetsSmearedUp=process.slimmedJetsSmeared.clone(variation=cms.int32(1))
process.p+=process.slimmedJetsSmearedUp

# example for cumputing statistical and systematic uncertainties on jet resolution separately
#for source in ["Stat","Jec","Gaus","Rest","Time"]:
#  setattr(process,"slimmedJetsSmeared"+source+"Down",process.slimmedJetsSmeared.clone(uncertaintySource=source,variation=cms.int32(-1)))
#  process.p+=getattr(process,"slimmedJetsSmeared"+source+"Down")
#  setattr(process,"slimmedJetsSmeared"+source+"Up",process.slimmedJetsSmeared.clone(uncertaintySource=source,variation=cms.int32(1)))
#  process.p+=getattr(process,"slimmedJetsSmeared"+source+"Up")

process.schedule=cms.Schedule(process.p,process.MINIAODSIMoutput_step)
