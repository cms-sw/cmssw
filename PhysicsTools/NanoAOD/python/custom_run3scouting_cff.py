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

gtStage2Digis.InputLabel = cms.InputTag("hltFEDSelectorL1")
triggerTask = cms.Task(unpackedPatTrigger,triggerObjectTable,l1bits)
triggerSequence = cms.Sequence(L1TRawToDigi+patTrigger+selectedPatTrigger+slimmedPatTrigger+cms.Sequence(triggerTask))

# MC tasks
genJetTask = cms.Task(ak4ScoutingJetMatchGen,ak4ScoutingJetExtTable,ak8ScoutingJetMatchGen,ak8ScoutingJetExtTable)
puTask = cms.Task(puTable)

nanoTableTaskCommon = cms.Task(photonScoutingTable,muonScoutingTable,electronScoutingTable,trackScoutingTable,primaryvertexScoutingTable,displacedvertexScoutingTable,rhoScoutingTable,metScoutingTable,particleTask,particleTableTask,ak4JetTableTask,ak8JetTableTask)

nanoSequenceCommon = cms.Sequence(triggerSequence,nanoTableTaskCommon)

nanoSequence = cms.Sequence(nanoSequenceCommon)

nanoSequenceMC = cms.Sequence(nanoSequenceCommon + cms.Sequence(cms.Task(genJetTask,puTask)))

def nanoAOD_customizeCommon(process):
    return process
