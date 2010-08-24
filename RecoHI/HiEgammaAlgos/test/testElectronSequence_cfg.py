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
'/store/relval/CMSSW_3_8_1/RelValPyquen_ZeemumuJets_pt10_2760GeV/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V8-v1/0013/F4EFE636-BFA3-DF11-B23A-001A92811742.root',
'/store/relval/CMSSW_3_8_1/RelValPyquen_ZeemumuJets_pt10_2760GeV/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V8-v1/0013/C6C4B6A7-BEA3-DF11-90CF-001A92811748.root',
'/store/relval/CMSSW_3_8_1/RelValPyquen_ZeemumuJets_pt10_2760GeV/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V8-v1/0013/C464D75B-BFA3-DF11-97E0-00304867916E.root',
'/store/relval/CMSSW_3_8_1/RelValPyquen_ZeemumuJets_pt10_2760GeV/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V8-v1/0013/B6880265-C0A3-DF11-A8AD-003048678B72.root',
'/store/relval/CMSSW_3_8_1/RelValPyquen_ZeemumuJets_pt10_2760GeV/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V8-v1/0013/A82B0A5B-BFA3-DF11-BF15-003048678B7E.root',
'/store/relval/CMSSW_3_8_1/RelValPyquen_ZeemumuJets_pt10_2760GeV/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V8-v1/0013/988C704E-BFA3-DF11-9586-002354EF3BDE.root',
'/store/relval/CMSSW_3_8_1/RelValPyquen_ZeemumuJets_pt10_2760GeV/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V8-v1/0013/5CB50765-C0A3-DF11-84B6-003048678B72.root',
'/store/relval/CMSSW_3_8_1/RelValPyquen_ZeemumuJets_pt10_2760GeV/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V8-v1/0013/2830B462-C0A3-DF11-A942-001A92810AE0.root',
'/store/relval/CMSSW_3_8_1/RelValPyquen_ZeemumuJets_pt10_2760GeV/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V8-v1/0013/1C7A1E65-C0A3-DF11-A16B-003048678B72.root' ] 
options.maxEvents = 1 

# get and parse the command line arguments
options.parseArguments()


##################################################################################
# Some Services
	   
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

# GenFilter for opposite-sign status=1 electrons from the embedded signal within the acceptance
process.eegenfilter = cms.EDFilter("MCParticlePairFilter",
    moduleLabel = cms.untracked.string("hiSignal"),                               
    Status = cms.untracked.vint32(1, 1),
    MinPt = cms.untracked.vdouble(2.5, 2.5),
    MaxEta = cms.untracked.vdouble(2.5, 2.5),
    MinEta = cms.untracked.vdouble(-2.5, -2.5),
    ParticleCharge = cms.untracked.int32(-1),
    ParticleID1 = cms.untracked.vint32(11),
    ParticleID2 = cms.untracked.vint32(11)
)

##################################################################################
#Reconstruction			
process.load("Configuration.StandardSequences.RawToDigi_cff")		    # RawToDigi
process.load("Configuration.StandardSequences.ReconstructionHeavyIons_cff") # full heavy ion reconstruction
process.load("RecoHI.HiEgammaAlgos.HiElectronSequence_cff")                 # gsf electrons

##############################################################################
# Output EDM File
process.load("Configuration.EventContent.EventContentHeavyIons_cff")        #load keep/drop output commands
process.output = cms.OutputModule("PoolOutputModule",
                                  process.FEVTDEBUGEventContent,
                                  fileName = cms.untracked.string(options.output),
                                  SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('filter_step'))
                                  )
process.output.outputCommands.extend(["keep *_*_*_TEST"])

##################################################################################

# Paths
process.filter_step = cms.Path(process.eegenfilter)

process.reco_step = cms.Path(process.eegenfilter
                             * process.RawToDigi
                             * process.reconstructionHeavyIons
                             * process.hiElectronSequence)

process.out_step = cms.EndPath(process.output)

process.schedule = cms.Schedule(process.filter_step,process.reco_step,process.out_step)
