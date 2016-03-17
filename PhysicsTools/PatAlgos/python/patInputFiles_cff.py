import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.cmsswVersionTools import pickRelValInputFiles

# /RelValTTbar_13/MINIAODSIM/PU25ns_80X_mcRun2_asymptotic_v4-v1
filesRelValTTbarPileUpMINIAODSIM = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_8_0_0_pre6'
                        #, relVal        = 'RelValTTbar_13'
                        #, globalTag     = 'PU25ns_80X_mcRun2_asymptotic_v4-v1'
                        #, dataTier      = 'MINIAODSIM'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_8_0_0_pre6/RelValTTbar_13/MINIAODSIM/PU25ns_80X_mcRun2_asymptotic_v4-v1/10000/F8714A00-7AD0-E511-9EBC-0025905A613C.root'
    )

# /RelValTTbar_13/GEN-SIM-RECO/80X_mcRun2_asymptotic_v4-v1 [AODSIM does not exist (anymore?)]
filesRelValProdTTbarAODSIM = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_8_0_0_pre6'
                        #, relVal        = 'RelValProdTTbar_13'
                        #, globalTag     = '80X_mcRun2_asymptotic_v4-v1'
                        #, dataTier      = 'GEN-SIM-RECO'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_8_0_0_pre6/RelValTTbar_13/GEN-SIM-RECO/80X_mcRun2_asymptotic_v4-v1/10000/D62B8277-E2D0-E511-AC66-0CC47A4D7654.root'
    )

# RelValTTbar_13/GEN-SIM-RECO/80X_mcRun2_asymptotic_v4-v1
filesRelValTTbarGENSIMRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_8_0_0_pre6'
                        #, relVal        = 'RelValTTbar_13'
                        #, globalTag     = '80X_mcRun2_asymptotic_v4-v1'
                        #, dataTier      = 'GEN-SIM-RECO'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_8_0_0_pre6/RelValTTbar_13/GEN-SIM-RECO/80X_mcRun2_asymptotic_v4-v1/10000/D62B8277-E2D0-E511-AC66-0CC47A4D7654.root'
    )

# /RelValTTbar_13/CMSSW_7_6_0_pre4-PU25ns_76X_mcRun2_asymptotic_v1_FastSim-v1/GEN-SIM-DIGI-RECO
filesRelValTTbarPileUpFastSimGENSIMDIGIRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_6_0_pre4'
                        #, relVal        = 'RelValTTbar_13'
                        #, globalTag     = 'PU25ns_76X_mcRun2_asymptotic_v1_FastSim'
                        #, dataTier      = 'GEN-SIM-DIGI-RECO'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        #'/store/relval/CMSSW_7_6_0_pre4/RelValTTbar_13/GEN-SIM-DIGI-RECO/PU25ns_76X_mcRun2_asymptotic_v1_FastSim-v1/00000/0026DA13-BF4F-E511-A342-0025905A6088.root'
    # temporary: produce fastsim sample on the fly
    # can be restored as soon as relval samples are available with the new fastsim rechits
    "file:ttbarForFastSimTest.root"
    )

# /RelValTTbar_13/CMSSW_7_6_0_pre7-PU25ns_76X_mcRun2_asymptotic_v5-v1/GEN-SIM-RECO
filesRelValTTbarPileUpGENSIMRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_8_0_0_pre6'
                        #, relVal        = 'RelValTTbar_13'
                        #, globalTag     = 'PU25ns_80X_mcRun2_asymptotic_v4-v1'
                        #, dataTier      = 'GEN-SIM-RECO'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_8_0_0_pre6/RelValTTbar_13/GEN-SIM-RECO/PU25ns_80X_mcRun2_asymptotic_v4-v1/10000/D8D11207-7AD0-E511-B82A-0025905A607A.root'
    )

# /SingleMu/CMSSW_7_6_0_pre7-76X_dataRun1_HLT_frozen_v4_RelVal_mu2012D-v1/MINIAOD
filesRelValSingleMuMINIAOD = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_6_0_pre7'
                        #, relVal        = 'SingleMu'
                        #, globalTag     = '76X_dataRun1_HLT_frozen_v4_RelVal_mu2012D'
                        #, dataTier      = 'MINIAOD'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_6_0_pre7/SingleMu/MINIAOD/76X_dataRun1_HLT_frozen_v4_RelVal_mu2012D-v1/00000/04E2CB97-6172-E511-B58F-002590596486.root'
    )

# /SingleMu/CMSSW_7_6_0_pre7-76X_dataRun1_HLT_frozen_v4_RelVal_mu2012D-v1/RECO
filesSingleMuRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_6_0_pre7'
                        #, relVal        = 'SingleMu'
                        #, dataTier      = 'RECO'
                        #, globalTag     = '76X_dataRun1_HLT_frozen_v4_RelVal_mu2012D'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_6_0_pre7/SingleMu/RECO/76X_dataRun1_HLT_frozen_v4_RelVal_mu2012D-v1/00000/000026F5-2172-E511-B57C-0025905B8582.root'
    )
