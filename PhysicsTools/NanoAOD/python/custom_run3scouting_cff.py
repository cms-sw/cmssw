import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.run3scouting_cff import *
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

# from 2024, there are two muon collections (https://its.cern.ch/jira/browse/CMSHLT-3089)
run3_scouting_nanoAOD_2024.toReplaceWith(scoutingMuonTableTask, cms.Task(scoutingMuonVtxTable, scoutingMuonNoVtxTable))\
    .toReplaceWith(scoutingMuonDisplacedVertexTableTask, cms.Task(scoutingMuonVtxDisplacedVertexTable, scoutingMuonNoVtxDisplacedVertexTable))

# Scouting Electron
scoutingElectronTableTask = cms.Task(scoutingElectronTable)

# from 2023, scouting electron's tracks are added as std::vector since multiple tracks can be associated to a scouting electron
# plugin to select the best track to reduce to a single track per scouting electron is added
(run3_scouting_nanoAOD_2023 | run3_scouting_nanoAOD_2024).toReplaceWith(
     scoutingElectronTableTask, cms.Task(scoutingElectronBestTrack, scoutingElectronTable)
)

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
    scoutingFatPFJetReclusterGlobalParticleTransformerJetTagInfos, scoutingFatPFJetReclusterGlobalParticleTransformerJetTags, # jet tagging with Global Particle Transformer
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
    scoutingNanoTaskCommon.add(scoutingElectronTableTask)
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
scoutingTriggerSequence = cms.Sequence(scoutingTriggerTask)
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

# additional customisation for running with ScoutingPFMonitor/RAW inputs
# should be used with default customiseScoutingNano
# this is suitable when ScoutingPFMonitor/RAW is involved, e.g. RAW, RAW-MiniAOD two-file solution, full chain RAW-MiniAOD-NanoAOD
# when running full chain RAW-MiniAOD-NanoAOD, this ensures that gtStage2Digis, gmtStage2Digis, and caloStage2Digis are run
def customiseScoutingNanoForScoutingPFMonitor(process):
    process = skipEventsWithoutScouting(process)

    # replace gtStage2DigisScouting with standard gtStage2Digis
    process.scoutingTriggerTask.remove(process.gtStage2DigisScouting)
    process.scoutingTriggerTask.add(process.gtStage2Digis)

    # add gmtStage2Digis
    process.load("EventFilter.L1TRawToDigi.gmtStage2Digis_cfi")
    process.scoutingTriggerTask.add(process.gmtStage2Digis)

    # add caloStage2Digis
    process.load("EventFilter.L1TRawToDigi.caloStage2Digis_cfi")
    process.scoutingTriggerTask.add(process.caloStage2Digis)

    # replace l1bitsScouting with standard l1bits
    process.scoutingTriggerTask.remove(process.l1bitsScouting)
    process.scoutingTriggerTask.add(process.l1bits)

    # change src for l1 objects
    process.l1MuScoutingTable.src = cms.InputTag("gmtStage2Digis", "Muon")
    process.l1EGScoutingTable.src = cms.InputTag("caloStage2Digis", "EGamma")
    process.l1TauScoutingTable.src = cms.InputTag("caloStage2Digis", "Tau")
    process.l1JetScoutingTable.src = cms.InputTag("caloStage2Digis", "Jet")
    process.l1EtSumScoutingTable.src = cms.InputTag("caloStage2Digis", "EtSum")

    return process

# additional customisation for running with ScoutingPFMonitor/MiniAOD inputs alone
# can also be used on MC input
# should be used with default customiseScoutingNano and NOT with customiseScoutingNanoForScoutingPFMonitor
def customiseScoutingNanoFromMini(process):
    # when running on data, assume ScoutingPFMonitor/MiniAOD dataset as inputs
    runOnData = hasattr(process,"NANOAODSIMoutput") or hasattr(process,"NANOAODoutput")
    if runOnData:
        process = skipEventsWithoutScouting(process)

    # remove gtStage2Digis since they are already run for Mini
    process.scoutingTriggerTask.remove(process.gtStage2DigisScouting)

    # replace l1bitsScouting with standard l1bits
    process.scoutingTriggerTask.remove(process.l1bitsScouting)
    process.scoutingTriggerTask.add(process.l1bits)

    # change src for l1 objects
    process.l1MuScoutingTable.src = cms.InputTag("gmtStage2Digis", "Muon")
    process.l1EGScoutingTable.src = cms.InputTag("caloStage2Digis", "EGamma")
    process.l1TauScoutingTable.src = cms.InputTag("caloStage2Digis", "Tau")
    process.l1JetScoutingTable.src = cms.InputTag("caloStage2Digis", "Jet")
    process.l1EtSumScoutingTable.src = cms.InputTag("caloStage2Digis", "EtSum")

    return process

# skip events without scouting object products
# this may be needed since for there are some events which do not contain scouting object products in 2022-24
# this is fixed for 2025: https://its.cern.ch/jira/browse/CMSHLT-3331
def skipEventsWithoutScouting(process):
    # if scouting paths are triggered, scouting objects will be reconstructed
    # so we select events passing scouting paths
    import HLTrigger.HLTfilters.hltHighLevel_cfi

    process.scoutingTriggerPathFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
            HLTPaths = cms.vstring("Dataset_ScoutingPFRun3")
            )

    process.nanoSkim_step = cms.Path(process.scoutingTriggerPathFilter)
    process.schedule.extend([process.nanoSkim_step])

    if hasattr(process, "NANOAODoutput"):
        process.NANOAODoutput.SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring("nanoSkim_step"))

    if hasattr(process, "NANOAODEDMoutput"):
        process.NANOEDMAODoutput.SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring("nanoSkim_step"))

    if hasattr(process, "write_NANOAOD"): # PromptReco
        process.write_NANOAOD.SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring("nanoSkim_step")) 

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

# this adds all electron tracks in addition to best track selected
def addScoutingElectronTrack(process):
    process.scoutingElectronTable.externalVariables.bestTrack_index\
            = ExtVar(cms.InputTag("scoutingElectronBestTrack", "Run3ScoutingElectronBestTrackIndex"), int, doc="best track index")

    process.scoutingElectronTable.collectionVariables = cms.PSet(
        ScoutingElectronTrack = cms.PSet(
            name = cms.string("ScoutingElectronTrack"),
            doc = cms.string("Scouting Electron Tracks"),
            useCount = cms.bool(True),
            useOffset = cms.bool(True),
            variables = cms.PSet(
                d0 = Var("trkd0", "float", doc="track d0"),
                dz = Var("trkdz", "float", doc="track dz"),
                pt = Var("trkpt", "float", doc="track pt"),
                eta = Var("trketa", "float", doc="track eta"),
                phi = Var("trkphi", "float", doc="track phi"),
                pMode = Var("trkpMode", "float", doc="track pMode"),
                etaMode = Var("trketaMode", "float", doc="track etaMode"),
                phiMode = Var("trkphiMode", "float", doc="track phiMode"),
                qoverpModeError = Var("trkqoverpModeError", "float", doc="track qoverpModeError"),
                chi2overndf = Var("trkchi2overndf", "float", doc="track normalized chi squared"),
                charge = Var("trkcharge", "int", doc="track charge"),
            ),
        ),
    )
    return process
