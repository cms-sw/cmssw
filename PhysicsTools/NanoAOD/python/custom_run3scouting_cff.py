import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.run3scouting_cff import *
from L1Trigger.Configuration.L1TRawToDigi_cff import *
from EventFilter.L1TRawToDigi.gtStage2Digis_cfi import gtStage2Digis
from PhysicsTools.NanoAOD.triggerObjects_cff import l1bits
from PhysicsTools.NanoAOD.globals_cff import puTable

############################
### Sub Task Definitions ###
############################

# Task contains all dependent tasks
# ExtensionTask must be run on top of another Task

#############################
# Scouting Original Objects #
#############################

# Scouting Muon
scoutingMuonTableTask = cms.Task(scoutingMuonTable)
scoutingMuonDisplacedVertexTableTask = cms.Task(scoutingMuonDisplacedVertexTable)

# from 2024, there are two muon collections
from Configuration.Eras.Modifier_run3_scouting_nanoAOD_post2023_cff import run3_scouting_nanoAOD_post2023
run3_scouting_nanoAOD_post2023.toReplaceWith(scoutingMuonTableTask, cms.Task(scoutingMuonVtxTable, scoutingMuonNoVtxTable))\
    .toReplaceWith(scoutingMuonDisplacedVertexTableTask, cms.Task(scoutingMuonVtxDisplacedVertexTable, scoutingMuonNoVtxDisplacedVertexTable))

# other collections are directly from original Run3Scouting objects, so unnessary to define tasks

############################
# Scouting Derived Objects #
############################

scoutingPFCandidateTask = cms.Task(scoutingPFCandidate, scoutingPFCandidateTable)
scoutingPFJetReclusterTask = cms.Task(
    scoutingPFCandidate, # translate to reco::PFCandidate, used as input
    scoutingPFJetRecluster, # jet clustering
    scoutingPFJetReclusterParticleNetJetTagInfos, scoutingPFJetReclusterParticleNetJetTags, # jet tagging
    scoutingPFJetReclusterTable
)
scoutingPFJetReclusterMatchGenExtensionTask = cms.Task(
    scoutingPFJetReclusterMatchGen, # gen jet matching
    scoutingPFJetReclusterMatchGenExtensionTable
)

scoutingFatPFJetReclusterTask = cms.Task(
    scoutingPFCandidate, # translate to reco::PFCandidate, used as input
    scoutingFatPFJetRecluster, # jet clustering
    scoutingFatPFJetReclusterParticleNetJetTagInfos, scoutingFatPFJetReclusterParticleNetJetTags, # jet tagging
    scoutingFatPFJetReclusterSoftDrop, scoutingFatPFJetReclusterSoftDropMass, # softdrop mass
    scoutingFatPFJetReclusterParticleNetJetTagInfos, scoutingFatPFJetReclusterParticleNetMassRegressionJetTags, # regressed mass
    scoutingFatPFJetReclusterEcfNbeta1, scoutingFatPFJetReclusterNjettiness, # substructure variables
    scoutingFatPFJetReclusterTable
)
scoutingFatPFJetReclusterMatchGenExtensionTask = cms.Task(
    scoutingFatPFJetReclusterMatchGen, # gen jet matching
    scoutingFatPFJetReclusterMatchGenExtensionTable
)

############################
# Trigger Bits and Objects #
############################

## L1 decisions
gtStage2DigisScouting = gtStage2Digis.clone(InputLabel="hltFEDSelectorL1")
l1bitsScouting = l1bits.clone(src="gtStage2DigisScouting") 

## L1 objects
from PhysicsTools.NanoAOD.l1trig_cff import *
l1MuScoutingTable = l1MuTable.clone(src=cms.InputTag("gtStage2DigisScouting", "Muon"))
l1EGScoutingTable = l1EGTable.clone(src=cms.InputTag("gtStage2DigisScouting", "EGamma"))
l1TauScoutingTable = l1TauTable.clone(src=cms.InputTag("gtStage2DigisScouting", "Tau"))
l1JetScoutingTable = l1JetTable.clone(src=cms.InputTag("gtStage2DigisScouting", "Jet"))
l1EtSumScoutingTable = l1EtSumTable.clone(src=cms.InputTag("gtStage2DigisScouting", "EtSum"))

# reduce the variables to the core variables as only these are available in gtStage2Digis
l1MuScoutingTable.variables = cms.PSet(l1MuonReducedVars)
l1EGScoutingTable.variables = cms.PSet(l1EGReducedVars)
l1TauScoutingTable.variables = cms.PSet(l1TauReducedVars)
l1JetScoutingTable.variables = cms.PSet(l1JetReducedVars)
l1EtSumScoutingTable.variables = cms.PSet(l1EtSumReducedVars)

##############################
### Main Tasks Definitions ###
##############################

# default configuration for ScoutingNano common for both data and MC
def prepareScoutingNanoTaskCommon():
    # Scouting original objects
    # all scouting objects are saved except PF Candidate and Track
    scoutingNanoTaskCommon = cms.Task()
    scoutingNanoTaskCommon.add(scoutingMuonTableTask, scoutingMuonDisplacedVertexTableTask)
    scoutingNanoTaskCommon.add(scoutingElectronTable)
    scoutingNanoTaskCommon.add(scoutingPhotonTable)
    scoutingNanoTaskCommon.add(scoutingPrimaryVertexTable)
    scoutingNanoTaskCommon.add(scoutingPFJetTable)
    scoutingNanoTaskCommon.add(scoutingMETTable, scoutingRhoTable)
    
    # Scouting derived objects
    scoutingNanoTaskCommon.add(scoutingPFJetReclusterTask)
    scoutingNanoTaskCommon.add(scoutingFatPFJetReclusterTask)

    return scoutingNanoTaskCommon

# tasks related to trigger bits and objects
def prepareScoutingTriggerTask():
    scoutingTriggerTask = cms.Task(gtStage2DigisScouting, l1bitsScouting)
    scoutingTriggerTask.add(cms.Task(l1MuScoutingTable, l1EGScoutingTable, l1TauScoutingTable, l1JetScoutingTable, l1EtSumScoutingTable))
    return scoutingTriggerTask

# additional tasks for running on MC
def prepareScoutingNanoTaskMC():
    scoutingNanoTaskMC = cms.Task()
    scoutingNanoTaskMC.add(scoutingPFJetReclusterMatchGenExtensionTask)
    scoutingNanoTaskMC.add(scoutingFatPFJetReclusterMatchGenExtensionTask)

    scoutingNanoTaskMC.add(puTable)
    return scoutingNanoTaskMC

# Common tasks added to main scoutingNanoSequence
scoutingNanoTaskCommon = prepareScoutingNanoTaskCommon()
scoutingNanoSequence = cms.Sequence(scoutingNanoTaskCommon)

# Specific tasks which will be added to sequence during customization
scoutingTriggerTask = prepareScoutingTriggerTask()
scoutingTriggerSequence = cms.Sequence(L1TRawToDigi+cms.Sequence(scoutingTriggerTask))
scoutingNanoTaskMC = prepareScoutingNanoTaskMC()

def customiseScoutingNano(process):
    # if running with standard NanoAOD, triggerSequence is already added
    # if running standalone, triggerSequence need to be added
    if not ((hasattr(process, "nanoSequence") and process.schedule.contains(process.nanoSequence))
            or hasattr(process, "nanoSequenceMC") and process.schedule.contains(process.nanoSequenceMC)):
        process.trigger_step = cms.Path(process.scoutingTriggerSequence)
        process.schedule.extend([process.trigger_step])

    # specific tasks when running on MC
    runOnMC = hasattr(process,"NANOEDMAODSIMoutput") or hasattr(process,"NANOAODSIMoutput")
    if runOnMC:
        process.scoutingNanoSequence.associate(scoutingNanoTaskMC)
    
    return process

#####################
### Customisation ###
#####################
# these function are designed to be used with --customise flag in cmsDriver.py
# e.g. --customise PhysicsTools/NanoAOD/python/custom_run3scouting_cff.addScoutingPFCandidate

# reconfigure for running with ScoutingPFMonitor/MiniAOD inputs alone
# should be used with default customiseScoutingNano
def customiseScoutingNanoFromMini(process):
    # remove L1TRawToDigi
    process.scoutingTriggerSequence.remove(process.L1TRawToDigi)

    # remove gtStage2Digis since they are already run for Mini
    process.scoutingTriggerTask.remove(process.gtStage2DigisScouting)

    # change src for l1 bits
    process.l1bitsScouting.src = cms.InputTag("gtStage2Digis")

    # change src for l1 objects
    process.l1MuScoutingTable.src = cms.InputTag("gmtStage2Digis", "Muon")
    process.l1EGScoutingTable.src = cms.InputTag("caloStage2Digis", "EGamma")
    process.l1TauScoutingTable.src = cms.InputTag("caloStage2Digis", "Tau")
    process.l1JetScoutingTable.src = cms.InputTag("caloStage2Digis", "Jet")
    process.l1EtSumScoutingTable.src = cms.InputTag("caloStage2Digis", "EtSum")

    return process

def addScoutingTrack(process):
    process.scoutingNanoSequence.associate(cms.Task(scoutingTrackTable))
    return process

def addScoutingParticle(process):
    # original PF candidate without post-processing
    process.scoutingNanoSequence.associate(cms.Task(scoutingParticleTable))
    return process

def addScoutingPFCandidate(process):
    # PF candidate after translation to reco::PFCandidate
    process.scoutingNanoSequence.associate(scoutingPFCandidateTask)
    return process
