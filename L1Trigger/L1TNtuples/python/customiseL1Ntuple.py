import FWCore.ParameterSet.Config as cms
import os

##############################################################################
# customisations for L1 ntuple generation
#
# Add new customisations to this file!
#
# Example usage :
#   cmsDriver.py testNtuple -s NONE --customise=L1Trigger/L1TNtuples/customiseL1Ntuple.customiseL1NtupleAOD --conditions=auto:run2_mc_50ns --filein='/store/relval/CMSSW_7_5_0_pre1/RelValProdTTbar_13/AODSIM/MCRUN2_74_V7-v1/00000/48159643-5EE3-E411-818F-0025905A48F0.root' -n 100
#
##############################################################################

def L1NtupleAOD(process):

    process.TFileService = cms.Service("TFileService",
                                       fileName = cms.string('L1NtupleAOD.root')
    )

    process.load('L1Trigger.L1TNtuples.L1NtupleAOD_cff')
    process.l1ntupleaod = cms.Path(
        process.L1NtupleAOD
    )

    process.schedule.append(process.l1ntupleaod)

    return process

def L1NtupleRAW(process):

    process.TFileService = cms.Service("TFileService",
                                       fileName = cms.string('L1NtupleRAW.root')
    )

    process.load('L1Trigger.L1TNtuples.L1NtupleRAW_cff')
    process.l1ntupleraw = cms.Path(
        process.L1NtupleRAW
    )

    process.schedule.append(process.l1ntupleraw)

    return process
