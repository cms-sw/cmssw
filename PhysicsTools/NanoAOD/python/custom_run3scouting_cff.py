import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.run3scouting_cff import *
from PhysicsTools.NanoAOD.globals_cff import puTable
from PhysicsTools.NanoAOD.triggerObjects_cff import unpackedPatTrigger, triggerObjectTable, l1bits
from L1Trigger.Configuration.L1TRawToDigi_cff import *
from EventFilter.L1TRawToDigi.gtStage2Digis_cfi import gtStage2Digis
from PhysicsTools.PatAlgos.triggerLayer1.triggerProducer_cfi import patTrigger
from PhysicsTools.PatAlgos.slimming.selectedPatTrigger_cfi import selectedPatTrigger
from PhysicsTools.PatAlgos.slimming.slimmedPatTrigger_cfi import slimmedPatTrigger

# common tasks
particleTask = cms.Task(scoutingPFCands)
particleTableTask = cms.Task(particleScoutingTable)
ak4JetTableTask = cms.Task(ak4ScoutingJets,ak4ScoutingJetParticleNetJetTagInfos,ak4ScoutingJetParticleNetJetTags,ak4ScoutingJetTable)
ak8JetTableTask = cms.Task(ak8ScoutingJets,ak8ScoutingJetsSoftDrop,ak8ScoutingJetsSoftDropMass,ak8ScoutingJetEcfNbeta1,ak8ScoutingJetNjettiness,ak8ScoutingJetParticleNetJetTagInfos,ak8ScoutingJetParticleNetJetTags,ak8ScoutingJetParticleNetMassRegressionJetTags,ak8ScoutingJetTable)

## L1 decisions
gtStage2DigisScouting = gtStage2Digis.clone(InputLabel="hltFEDSelectorL1")
l1bitsScouting = l1bits.clone(src="gtStage2DigisScouting")
patTriggerScouting = patTrigger.clone(l1tAlgBlkInputTag="gtStage2DigisScouting",l1tExtBlkInputTag="gtStage2DigisScouting")

## L1 objects
from PhysicsTools.NanoAOD.l1trig_cff import *
l1MuScoutingTable = l1MuTable.clone(src=cms.InputTag("gtStage2DigisScouting","Muon"))
l1JetScoutingTable = l1JetTable.clone(src=cms.InputTag("gtStage2DigisScouting","Jet"))
l1EGScoutingTable = l1EGTable.clone(src=cms.InputTag("gtStage2DigisScouting","EGamma"))
l1TauScoutingTable = l1TauTable.clone(src=cms.InputTag("gtStage2DigisScouting","Tau"))
l1EtSumScoutingTable = l1EtSumTable.clone(src=cms.InputTag("gtStage2DigisScouting","EtSum"))

#reduce the variables to the core variables as only these are available in gtStage2Digis
l1EGScoutingTable.variables = cms.PSet(l1EGReducedVars)
l1MuScoutingTable.variables = cms.PSet(l1MuonReducedVars)
l1JetScoutingTable.variables = cms.PSet(l1JetReducedVars)
l1TauScoutingTable.variables = cms.PSet(l1TauReducedVars)
l1EtSumScoutingTable.variables = cms.PSet(l1EtSumReducedVars)

selectedPatTriggerScouting = selectedPatTrigger.clone(src="patTriggerScouting")
slimmedPatTriggerScouting = slimmedPatTrigger.clone(src="selectedPatTriggerScouting")
unpackedPatTriggerScouting = unpackedPatTrigger.clone(patTriggerObjectsStandAlone="slimmedPatTriggerScouting")
triggerObjectTableScouting = triggerObjectTable.clone(src="unpackedPatTriggerScouting")

triggerTask = cms.Task(
    gtStage2DigisScouting, l1MuScoutingTable, l1EGScoutingTable, l1TauScoutingTable, l1JetScoutingTable, l1EtSumScoutingTable, 
    unpackedPatTriggerScouting,triggerObjectTableScouting,l1bitsScouting
)
triggerSequence = cms.Sequence(L1TRawToDigi+patTriggerScouting+selectedPatTriggerScouting+slimmedPatTriggerScouting+cms.Sequence(triggerTask))

# MC tasks
genJetTask = cms.Task(ak4ScoutingJetMatchGen,ak4ScoutingJetExtTable,ak8ScoutingJetMatchGen,ak8ScoutingJetExtTable)
puTask = cms.Task(puTable)

nanoTableTaskCommon = cms.Task(photonScoutingTable,muonScoutingTable,electronScoutingTable,trackScoutingTable,primaryvertexScoutingTable,displacedvertexScoutingTable,rhoScoutingTable,metScoutingTable,particleTask,particleTableTask,ak4JetTableTask,ak8JetTableTask)

nanoSequenceCommon = cms.Sequence(triggerSequence,nanoTableTaskCommon)

nanoSequence = cms.Sequence(nanoSequenceCommon)

nanoSequenceMC = cms.Sequence(nanoSequenceCommon + cms.Sequence(cms.Task(genJetTask,puTask)))

def nanoAOD_customizeCommon(process):
    return process
