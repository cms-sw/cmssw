import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.l1scoutingrun3_cff import *

#########################
# Default Configuration #
#########################

# since L1ScoutingNano should be run standalone only,
# this replace task and sequences in standard NanoAOD
nanoTableTaskCommon = cms.Task(
    l1scoutingMuonPhysicalValueMap,
    l1scoutingEGammaPhysicalValueMap,
    l1scoutingTauPhysicalValueMap,
    l1scoutingJetPhysicalValueMap,
    l1scoutingMuonTable,
    l1scoutingEGammaTable,
    l1scoutingTauTable,
    l1scoutingJetTable,
    l1scoutingEtSumTable,
    l1scoutingBMTFStubTable,
)

nanoSequenceCommon = cms.Sequence(l1scoutingJetPhysicalValueMap + cms.Sequence(nanoTableTaskCommon))

nanoSequence = cms.Sequence(nanoSequenceCommon)

nanoSequenceMC = cms.Sequence(nanoSequenceCommon)

def customiseL1ScoutingNanoAOD(process):
    # change OutputModule to OrbitNanoAODOutputModule
    if hasattr(process, "NANOAODoutput"):
        print("custom_l1scoutingrun3_cff: Change NANOAODoutput to OrbitNanoAODOutputModule")
        setattr(process, "NANOAODoutput",
            cms.OutputModule("OrbitNanoAODOutputModule",
                # copy from standard NanoAOD
                compressionAlgorithm = process.NANOAODoutput.compressionAlgorithm,
                compressionLevel = process.NANOAODoutput.compressionLevel,
                dataset = process.NANOAODoutput.dataset,
                fileName = process.NANOAODoutput.fileName,
                # change eventcontent
                outputCommands = cms.untracked.vstring(
                    "drop *",
                    "keep l1ScoutingRun3OrbitFlatTable_*_*_*"),
                # additional parameters for l1scouting 
                SelectEvents = cms.untracked.PSet(
                    SelectEvents = cms.vstring('nanoAOD_step') # l1scouting runs standalone only
                ),
                skipEmptyBXs = cms.bool(True), # drop empty bxs
            )
        )

    return process

#################
# Customisation #
#################
# these function are designed to be used with --customise flag in cmsDriver.py
# e.g. --customise PhysicsTools/NanoAOD/python/custom_l1scoutingrun3_cff.dropStub

# configure to run with L1ScoutingSelection dataset
# should be used with default customiseL1ScoutingNanoAOD
def customiseL1ScoutingNanoAODSelection(process):
    # change sources
    process.l1scoutingMuonPhysicalValueMap.src = cms.InputTag("FinalBxSelectorMuon", "Muon")
    process.l1scoutingEGammaPhysicalValueMap.src = cms.InputTag("FinalBxSelectorEGamma", "EGamma")
    process.l1scoutingJetPhysicalValueMap.src = cms.InputTag("FinalBxSelectorJet", "Jet")

    process.l1scoutingMuonTable.src = cms.InputTag("FinalBxSelectorMuon", "Muon")
    process.l1scoutingEGammaTable.src = cms.InputTag("FinalBxSelectorEGamma", "EGamma")
    process.l1scoutingJetTable.src = cms.InputTag("FinalBxSelectorJet", "Jet")
    process.l1scoutingEtSumTable.src = cms.InputTag("FinalBxSelectorBxSums", "EtSum")
    process.l1scoutingBMTFStubTable.src = cms.InputTag("FinalBxSelectorBMTFStub", "BMTFStub")

    # drop L1Tau
    process.nanoTableTaskCommon.remove(process.l1scoutingTauTable)

    # change parameters in OrbitNanoAODOutputModule
    process.NANOAODoutput.outputCommands += ["keep uints_*_SelBx_*"] # keep SelBx
    process.NANOAODoutput.selectedBx = cms.InputTag("FinalBxSelector", "SelBx") # use to select products

    return process

def addHardwareValues(process):
    # add hardware values to variables
    process.l1scoutingMuonTable.variables = cms.PSet(
            process.l1scoutingMuonTable.variables,
            l1scoutingMuonUnconvertedVariables
    )
    process.l1scoutingEGammaTable.variables = cms.PSet(
            process.l1scoutingEGammaTable.variables,
            l1scoutingCaloObjectUnconvertedVariables
    )
    process.l1scoutingTauTable.variables = cms.PSet(
            process.l1scoutingTauTable.variables,
            l1scoutingCaloObjectUnconvertedVariables
    )
    process.l1scoutingJetTable.variables = cms.PSet(
            process.l1scoutingJetTable.variables,
            l1scoutingCaloObjectUnconvertedVariables
    )

    # EtSum uses dedicated EDProducer and can add hardware values by setting a boolean
    process.l1scoutingEtSumTable.writeHardwareValues = cms.bool(True)

    return process

def keepHardwareValuesOnly(process):
    # first, add hardware values
    process = addHardwareValues(process)

    # remove physical values
    # currently external values are all physical values, so we can simple remove them
    process.l1scoutingMuonTable.externalVariables = cms.PSet()
    process.l1scoutingEGammaTable.externalVariables = cms.PSet()
    process.l1scoutingTauTable.externalVariables = cms.PSet()
    process.l1scoutingJetTable.externalVariables = cms.PSet()

    # EtSum uses dedicated EDProducer and can remove physical values by setting a boolean
    process.l1scoutingEtSumTable.writePhysicalValues = cms.bool(False)

    return process

def outputMultipleEtSums(process):
    process.l1scoutingEtSumTable.singleton = cms.bool(False)
    return process

def dropEmptyBXs(process):
    process.NANOAODoutput.skipEmptyBXs = cms.bool(True)
    return process

def keepEmptyBXs(process):
    process.NANOAODoutput.skipEmptyBXs = cms.bool(False)
    return process

def dropBMTFStub(process):
    process.nanoTableTaskCommon.remove(process.l1scoutingBMTFStubTable)
    return process
