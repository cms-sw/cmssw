### -------------------------------------------------------------------
### VarParsing allows one to specify certain parameters in the command line
### e.g.
### cmsRun testElectronSequence_cfg.py print maxEvents=10
### -------------------------------------------------------------------

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
process.GlobalTag.globaltag = 'MC_39Y_V2::All'

##################################################################################

# setup 'standard'  options
options = VarParsing.VarParsing ('standard')

# setup any defaults you want
options.output = 'test_out.root'
options.files = [
    '/store/relval/CMSSW_3_9_0/RelValPyquen_ZeemumuJets_pt10_2760GeV/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V2-v1/0052/C41EB5F8-1DD9-DF11-BDF9-003048678E92.root',
    '/store/relval/CMSSW_3_9_0/RelValPyquen_ZeemumuJets_pt10_2760GeV/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V2-v1/0052/C2B6CF74-1ED9-DF11-A35F-0018F3D0961A.root',
    '/store/relval/CMSSW_3_9_0/RelValPyquen_ZeemumuJets_pt10_2760GeV/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V2-v1/0052/A4EF86F3-1DD9-DF11-9B50-0026189438EA.root',
    '/store/relval/CMSSW_3_9_0/RelValPyquen_ZeemumuJets_pt10_2760GeV/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V2-v1/0052/84D2AFF7-1ED9-DF11-9945-00304867C0F6.root',
    '/store/relval/CMSSW_3_9_0/RelValPyquen_ZeemumuJets_pt10_2760GeV/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V2-v1/0052/80CC6AF2-1DD9-DF11-B89F-0030486790BA.root',
    '/store/relval/CMSSW_3_9_0/RelValPyquen_ZeemumuJets_pt10_2760GeV/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V2-v1/0052/70D088E0-22D9-DF11-A42C-003048679244.root',
    '/store/relval/CMSSW_3_9_0/RelValPyquen_ZeemumuJets_pt10_2760GeV/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V2-v1/0052/6AC7A36F-1ED9-DF11-9584-003048679296.root',
    '/store/relval/CMSSW_3_9_0/RelValPyquen_ZeemumuJets_pt10_2760GeV/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V2-v1/0052/54C85271-1ED9-DF11-9B1D-00261894395C.root',
    '/store/relval/CMSSW_3_9_0/RelValPyquen_ZeemumuJets_pt10_2760GeV/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V2-v1/0052/5207EDE9-1ED9-DF11-9BBB-00304867918A.root',
    '/store/relval/CMSSW_3_9_0/RelValPyquen_ZeemumuJets_pt10_2760GeV/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V2-v1/0052/4AFA1DFA-1AD9-DF11-BBEB-003048679046.root'
    ] 
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
