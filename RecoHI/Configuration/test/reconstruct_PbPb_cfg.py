import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("RECO")

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

#global tags for conditions data: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions#3XY_Releases_MC
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'MC_31X_V8::All'

##################################################################################
# Some services

process.SimpleMemoryCheck = cms.Service('SimpleMemoryCheck',
                                        ignoreTotal=cms.untracked.int32(0),
                                        oncePerEventMode = cms.untracked.bool(False)
                                        )
										
process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('towerMaker', 
        'caloTowers', 
        'iterativeConePu5CaloJets'),
    destinations = cms.untracked.vstring('cout', 
        'cerr'),
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
    fwkJobReports = cms.untracked.vstring('FrameworkJobReport.xml')
)

process.Timing = cms.Service("Timing")

##################################################################################

# setup 'standard'  options
options = VarParsing.VarParsing ('standard')

# setup any defaults you want
options.output = 'test_output_RECO.root'
#options.files = '/store/relval/CMSSW_3_3_0_pre1/RelValHydjetQ_MinBias_4TeV/GEN-SIM-RAW/MC_31X_V5-v1/0012/ECD0FB45-6796-DE11-B075-001D09F28D54.root'
options.files = '/store/relval/CMSSW_3_3_0_pre3/RelValHydjetQ_MinBias_4TeV/GEN-SIM-RAW/MC_31X_V8-v1/0015/DC571B73-43A1-DE11-BD0C-000423D98804.root'
options.maxEvents = 1 

# get and parse the command line arguments
options.parseArguments()

##################################################################################
# Input Source
process.source = cms.Source('PoolSource',
	fileNames = cms.untracked.vstring(options.files)
)
							
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents)
)

##################################################################################
# Digi + Reconstruction

process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.StandardSequences.ReconstructionHeavyIons_cff")

##############################################################################
# Output EDM File
process.load("Configuration.EventContent.EventContentHeavyIons_cff")
process.output = cms.OutputModule("PoolOutputModule",
    process.RECODEBUGEventContent,
    compressionLevel = cms.untracked.int32(2),
    commitInterval = cms.untracked.uint32(1),
    fileName = cms.untracked.string(options.output)
)

##################################################################################
# Paths
process.p = cms.Path(process.RawToDigi*process.reconstruct_PbPb)
process.outpath = cms.EndPath(process.output)


