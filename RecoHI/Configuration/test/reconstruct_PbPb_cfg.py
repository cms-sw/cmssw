import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("RECO")

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

#global tags for conditions data: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions#3XY_Releases_MC
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'MC_31X_V5::All'

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
#options.files= 'dcache:/pnfs/cmsaf.mit.edu/t2bat/cms/store/mc/Summer09/Hydjet_MinBias_4TeV/GEN-SIM-RAW/MC_31X_V3-GaussianVtx_312_ver1/0005/FECC5F18-1982-DE11-ACF9-001EC94BA3AE.root'
options.files= 'rfio:/castor/cern.ch/cms/store/relval/CMSSW_3_2_4/RelValHydjetQ_MinBias_4TeV/GEN-SIM-RAW/MC_31X_V3-v1/0010/D62F586C-4E84-DE11-80D3-000423D98E54.root'
#options.files= 'rfio:/castor/cern.ch/cms/store/relval/CMSSW_3_2_4/RelValHydjetQ_B0_4TeV/GEN-SIM-RAW/MC_31X_V3-v1/0010/FC70A0E1-6A84-DE11-AE66-000423D98DB4.root'
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
process.load("RecoHI.Configuration.Reconstruction_HI_cff")

##############################################################################
# Output EDM File
process.load("RecoHI.Configuration.RecoHI_EventContent_cff")
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


