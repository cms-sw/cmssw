# tool for testing the automatic assembly of
# DQM histogramming modules from the current MC HLT menu
# (tests code in HLTriggerOffline/Egamma/python/EgammaHLTValidationUtils.py)

#----------------------------------------------------------------------
# parameters
#----------------------------------------------------------------------

# the (single) path to run the test for
pathToTest = "HLT_Ele17_SW_TighterEleIdIsol_L1R_v3"


#----------------------------------------------------------------------

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

#----------------------------------------
# input files
#----------------------------------------
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(

                                # /RelValWE/CMSSW_3_10_0_pre7-START310_V1-v1/GEN-SIM-DIGI-RAW-HLTDEBUG
                                '/store/relval/CMSSW_3_10_0_pre7/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0103/B4107759-45FD-DF11-AABA-00261894396F.root',
                                '/store/relval/CMSSW_3_10_0_pre7/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/F47D670D-E0FC-DF11-89F6-0026189438A7.root',
                                '/store/relval/CMSSW_3_10_0_pre7/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/C2469357-DDFC-DF11-AA08-001A92810AA6.root',
                                '/store/relval/CMSSW_3_10_0_pre7/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/B8E45D58-DDFC-DF11-8ABD-0018F3D095EC.root',
                                '/store/relval/CMSSW_3_10_0_pre7/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/B6F3C564-DEFC-DF11-A322-001A92971BA0.root',
                                '/store/relval/CMSSW_3_10_0_pre7/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/8CDB4CD6-DCFC-DF11-A680-001A92971B0E.root',
                                '/store/relval/CMSSW_3_10_0_pre7/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/5AAA1057-DDFC-DF11-87A7-0018F3D096DC.root',
                                '/store/relval/CMSSW_3_10_0_pre7/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/2C664B9A-E0FC-DF11-A106-0026189438C9.root',
                                '/store/relval/CMSSW_3_10_0_pre7/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/10B11E80-DFFC-DF11-8B52-002618943924.root',
                                '/store/relval/CMSSW_3_10_0_pre7/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0100/8CEAC399-D4FC-DF11-BFE2-00304867BFA8.root',
                                '/store/relval/CMSSW_3_10_0_pre7/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0100/3A8A7EAA-D4FC-DF11-91AB-001A92971B48.root',
                                '/store/relval/CMSSW_3_10_0_pre7/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0100/06F9B0D3-D4FC-DF11-9B78-003048678FDE.root',
                                )
                            )

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

#----------------------------------------
# add the modules we want to test
# (producing the DQM histograms)
#----------------------------------------

import HLTriggerOffline.Egamma.EgammaHLTValidationUtils as EgammaHLTValidationUtils

# a 'reference' process to take (and analyze) the HLT menu from
refProcess = cms.Process("REF")

refProcess.load("HLTrigger.Configuration.HLT_GRun_cff")
process.dqmModule = EgammaHLTValidationUtils.EgammaDQMModuleMaker(refProcess, pathToTest,
                                                                   11, # type of generated particle
                                                                   1   # number of generated particles
                                                                   ).getResult()


del refProcess

process.dqmPath = cms.Path(

    # creates the sequence for requiring the number and type of generated particles
    EgammaHLTValidationUtils.makeGeneratedParticleAndFiducialVolumeFilter(process, 11, 1) *
    
    process.dqmModule)

#----------------------------------------
# E/gamma HLT specific DQM configuration
#----------------------------------------

process.post=cms.EDAnalyzer("EmDQMPostProcessor",
                            subDir = cms.untracked.string("HLT/HLTEgammaValidation"),
                            dataSet = cms.untracked.string("unknown"),
    )

#----------------------------------------
# add DQM configuration
#----------------------------------------
process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.p = cms.EndPath(process.post + process.dqmSaver)

# process.testW = cms.Path(process.egammaValidationSequence)

process.DQMStore.verbose = 0
process.DQM.collectorHost = ''
process.dqmSaver.convention = 'Online'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True

#----------------------------------------
