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
    process.l1ntuple = cms.Path(
        process.L1NtupleAOD
    )

    process.schedule.append(process.l1ntuple)

    return process

def L1NtupleRAW(process):

    process.TFileService = cms.Service("TFileService",
                                       fileName = cms.string('L1NtupleRAW.root')
    )

    process.load('L1Trigger.L1TNtuples.L1NtupleRAW_cff')
    process.l1ntuple = cms.Path(
        process.L1NtupleRAW
    )

    process.schedule.append(process.l1ntuple)

    # for 5 BX of candidates in L1Extra
    if process.producers.has_key("gctDigis"):
        process.gctDigis.numberOfGctSamplesToUnpack = cms.uint32(5)

    if process.producers.has_key("l1extraParticles"):
        process.l1extraParticles.centralBxOnly = cms.bool(False)

    return process

def L1NtupleAODRAW(process):


    # ---------------------------------------------------------------------
    # Set up electron ID (VID framework)
    # ---------------------------------------------------------------------
    from PhysicsTools.SelectorUtils.tools.vid_id_tools import *
    # turn on VID producer, indicate data format  to be
    # DataFormat.AOD or DataFormat.MiniAOD, as appropriate 
    dataFormat = DataFormat.AOD
    switchOnVIDElectronIdProducer(process, dataFormat)
    process.load("RecoEgamma.ElectronIdentification.egmGsfElectronIDs_cfi")
    from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry
    process.egmGsfElectronIDSequence = cms.Sequence(process.egmGsfElectronIDs)
    # define which IDs we want to produce
    idmod = 'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Spring15_25ns_V1_cff'  
    setupAllVIDIdsInModule(process,idmod,setupVIDElectronSelection)


    process.TFileService = cms.Service("TFileService",
                                       fileName = cms.string('L1NtupleAODRAW.root')
    )

    process.load('L1Trigger.L1TNtuples.L1NtupleRAW_cff')
    process.load('L1Trigger.L1TNtuples.L1NtupleAOD_cff')
    process.l1ntuple = cms.Path(
        process.L1NtupleRAW
        +process.egmGsfElectronIDSequence
        +process.L1NtupleAOD
    )

    process.schedule.append(process.l1ntuple)

    return process
