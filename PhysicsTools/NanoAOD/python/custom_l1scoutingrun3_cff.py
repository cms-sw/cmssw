import FWCore.ParameterSet.Config as cms

from PhysicsTools.NanoAOD.l1scoutingrun3_cff import *
from PhysicsTools.NanoAOD.L1SCOUTNanoAODEDMEventContent_cff import L1SCOUTNanoAODEDMEventContent, L1SCOUTNANOAODEventContent

from Configuration.Eras.Modifier_run3_l1scouting_2026_cff import run3_l1scouting_2026

l1scoutingNanoTask = cms.Task(
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

_l1scoutingNanoTask = l1scoutingNanoTask.copy()
_l1scoutingNanoTask.add(l1scoutingCaloTowerPhysicalValueMap)
_l1scoutingNanoTask.add(l1scoutingCaloTowerTable)
run3_l1scouting_2026.toReplaceWith(l1scoutingNanoTask, _l1scoutingNanoTask)

l1scoutingNanoSequence = cms.Sequence(l1scoutingNanoTask)

def _getNanoPoolOutputModuleLabels(process):
    """This method is meant to catch the instances of PoolOutputModule used for NanoAOD EDM products.
      - At present, these instances are identified by requiring the "dataset.dataTier" parameter (string)
        of the output module to contain the word "NANO" (this requirement is not case-sensitive).
    """
    ret = []
    for outModLabel, outMod in process.outputModules_().items():
        if outMod.type_() != "PoolOutputModule":
            continue
        try:
            if "nano" in outMod.dataset.dataTier.value().lower():
                ret += [outModLabel]
        except:
            pass
    return ret

def _getNanoAODOutputModuleLabels(process):
    return [outModLabel for outModLabel in process.outputModules_() \
        if process.outputModules_()[outModLabel].type_() == "NanoAODOutputModule"]

def _getOrbitNanoAODOutputModuleLabels(process):
    return [outModLabel for outModLabel in process.outputModules_() \
        if process.outputModules_()[outModLabel].type_() == "OrbitNanoAODOutputModule"]

def customiseL1ScoutingNanoAOD(process):
    """Customisation to run on the "L1Scouting" primary dataset
    """
    # NANOEDM: customise the event content of instances of PoolOutputModule
    for outModLabel in _getNanoPoolOutputModuleLabels(process):
        outMod = getattr(process, outModLabel)
        outMod.update_(L1SCOUTNanoAODEDMEventContent)

    # NANO: convert instances of NanoAODOutputModule to instances of OrbitNanoAODOutputModule
    #       (and customise the event content and compression settings)
    nanoAODOutputModuleLabels = _getNanoAODOutputModuleLabels(process)
    for outModLabel in nanoAODOutputModuleLabels:
        outMod = getattr(process, outModLabel)
        # customise the event content and compression settings
        outMod.update_(L1SCOUTNANOAODEventContent)
        # convert to OrbitNanoAODOutputModule
        setattr(process, outModLabel, cms.OutputModule("OrbitNanoAODOutputModule",
            **outMod.parameters_(),
            # skip BXs in which all the L1-Scouting tables are empty
            skipEmptyBXs = cms.bool(True),
            # "selectedBx" is not used if the InputTag's label is empty
            selectedBx = cms.InputTag("")
        ))

    return process

def customiseL1ScoutingNanoAODSelection(process):
    """Customisation to run on the "L1ScoutingSelection" primary dataset
    """
    process = customiseL1ScoutingNanoAOD(process)

    # change input collections from the L1SCOUT data tier
    process.l1scoutingMuonPhysicalValueMap.src = "FinalBxSelectorMuon:Muon"
    process.l1scoutingEGammaPhysicalValueMap.src = "FinalBxSelectorEGamma:EGamma"
    process.l1scoutingJetPhysicalValueMap.src = "FinalBxSelectorJet:Jet"
    process.l1scoutingCaloTowerPhysicalValueMap.src = "FinalBxSelectorCaloTower:CaloTower"

    process.l1scoutingMuonTable.src = "FinalBxSelectorMuon:Muon"
    process.l1scoutingEGammaTable.src = "FinalBxSelectorEGamma:EGamma"
    process.l1scoutingJetTable.src = "FinalBxSelectorJet:Jet"
    process.l1scoutingEtSumTable.src = "FinalBxSelectorBxSums:EtSum"
    process.l1scoutingBMTFStubTable.src = "FinalBxSelectorBMTFStub:BMTFStub"
    process.l1scoutingCaloTowerTable.src = "FinalBxSelectorCaloTower:CaloTower"

    # drop L1Tau
    process.l1scoutingNanoTask.remove(process.l1scoutingTauTable)

    # NANO: customise instances of OrbitNanoAODOutputModule
    for outModLabel in _getOrbitNanoAODOutputModuleLabels(process):
        outMod = getattr(process, outModLabel)
        # keep only the BXs in "selectedBx"
        outMod.selectedBx = "FinalBxSelector:SelBx"

    return process

###
### Additional customisations
###
###  - These functions are designed to be used with the --customise flag of cmsDriver.py,
###    e.g. "--customise PhysicsTools/NanoAOD/custom_l1scoutingrun3_cff.addHardwareValues".
###
def addHardwareValues(process):
    """Customisation to add the hardware values of L1-Scouting objects to the NanoAOD output tables
    """
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
    process.l1scoutingCaloTowerTable.variables = cms.PSet(
        process.l1scoutingCaloTowerTable.variables,
        l1scoutingCaloTowerUnconvertedVariables
    )

    # EtSum uses dedicated EDProducer and can add hardware values by setting a boolean
    process.l1scoutingEtSumTable.writeHardwareValues = True

    return process

def keepHardwareValuesOnly(process):
    """Customisation to keep ONLY the hardware values of L1-Scouting objects in the NanoAOD output tables
    """
    # first, add hardware values
    process = addHardwareValues(process)

    # remove physical values
    # currently external values are all physical values, so we can simple remove them
    process.l1scoutingMuonTable.externalVariables = cms.PSet()
    process.l1scoutingEGammaTable.externalVariables = cms.PSet()
    process.l1scoutingTauTable.externalVariables = cms.PSet()
    process.l1scoutingJetTable.externalVariables = cms.PSet()
    process.l1scoutingCaloTowerTable.externalVariables = cms.PSet()

    # EtSum uses dedicated EDProducer and can remove physical values by setting a boolean
    process.l1scoutingEtSumTable.writePhysicalValues = False

    return process

def outputMultipleEtSums(process):
    """Customisation to output multiple L1-Scouting EtSums in the relevant NanoAOD table
      (l1scoutingEtSumTable.singleton = False)
    """
    process.l1scoutingEtSumTable.singleton = False
    return process

def dropEmptyBXs(process):
    """Customisation to set "skipEmptyBXs = True" in all the instances of OrbitNanoAODOutputModule
    """
    for outModLabel in _getOrbitNanoAODOutputModuleLabels(process):
        getattr(process, outModLabel).skipEmptyBXs = True
    return process

def keepEmptyBXs(process):
    """Customisation to set "skipEmptyBXs = False" in all the instances of OrbitNanoAODOutputModule
    """
    for outModLabel in _getOrbitNanoAODOutputModuleLabels(process):
        getattr(process, outModLabel).skipEmptyBXs = False
    return process

def dropBMTFStub(process):
    """Customisation to remove the NanoAOD output table for the L1-Scouting BMTF stubs
    """
    process.l1scoutingNanoTask.remove(process.l1scoutingBMTFStubTable)
    return process
