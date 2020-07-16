import FWCore.ParameterSet.Config as cms

def UpdatePuppiTuneV14(process, runOnMC=True):
  #
  # Adapt for re-running PUPPI
  #
  print("customizePuppiTune_cff::UpdatePuppiTuneV14: Recomputing PUPPI with Tune v14, slimmedJetsPuppi and slimmedMETsPuppi")
  from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask, addToProcessAndTask
  task = getPatAlgosToolsTask(process)
  from PhysicsTools.PatAlgos.slimming.puppiForMET_cff import makePuppiesFromMiniAOD
  makePuppiesFromMiniAOD(process,True)
  process.puppi.useExistingWeights = False
  process.puppiNoLep.useExistingWeights = False
  from PhysicsTools.PatUtils.tools.runMETCorrectionsAndUncertainties import runMetCorAndUncFromMiniAOD
  runMetCorAndUncFromMiniAOD(process,isData=(not runOnMC),metType="Puppi",postfix="Puppi",jetFlavor="AK4PFPuppi",recoMetFromPFCs=True)
  from PhysicsTools.PatAlgos.patPuppiJetSpecificProducer_cfi import patPuppiJetSpecificProducer
  addToProcessAndTask('patPuppiJetSpecificProducer', patPuppiJetSpecificProducer.clone(src=cms.InputTag("patJetsPuppi")), process, task)
  from PhysicsTools.PatAlgos.tools.jetTools import updateJetCollection
  updateJetCollection(
     process,
     labelName = 'PuppiJetSpecific',
     jetSource = cms.InputTag('patJetsPuppi'),
  )
  process.updatedPatJetsPuppiJetSpecific.userData.userFloats.src = ['patPuppiJetSpecificProducer:puppiMultiplicity', 'patPuppiJetSpecificProducer:neutralPuppiMultiplicity', 'patPuppiJetSpecificProducer:neutralHadronPuppiMultiplicity', 'patPuppiJetSpecificProducer:photonPuppiMultiplicity', 'patPuppiJetSpecificProducer:HFHadronPuppiMultiplicity', 'patPuppiJetSpecificProducer:HFEMPuppiMultiplicity' ]
  addToProcessAndTask('slimmedJetsPuppi', process.updatedPatJetsPuppiJetSpecific.clone(), process, task)
  del process.updatedPatJetsPuppiJetSpecific
  process.puppiSequence = cms.Sequence(process.puppiMETSequence+process.fullPatMetSequencePuppi+process.patPuppiJetSpecificProducer+process.slimmedJetsPuppi)
  #
  # Adapt for PUPPI tune V14
  #
  process.puppi.PtMaxCharged = 20.
  process.puppi.EtaMinUseDeltaZ = 2.4
  process.puppi.PtMaxNeutralsStartSlope = 20.
  process.puppiNoLep.PtMaxCharged = 20.
  process.puppiNoLep.EtaMinUseDeltaZ = 2.4
  process.puppiNoLep.PtMaxNeutralsStartSlope = 20.

def UpdatePuppiTuneV14_MC(process):
  UpdatePuppiTuneV14(process,runOnMC=True)

def UpdatePuppiTuneV14_Data(process):
  UpdatePuppiTuneV14(process,runOnMC=False)
