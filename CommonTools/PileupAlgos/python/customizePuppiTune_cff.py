import FWCore.ParameterSet.Config as cms

def UpdatePuppiTuneV13(process):
  #
  # Adapt for re-running PUPPI
  #
  print("customizePuppiTune_cff::UpdatePuppiTuneV13: Recomputing PUPPI with Tune v13, slimmedJetsPuppi and slimmedMETsPuppi")
  from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask, addToProcessAndTask
  task = getPatAlgosToolsTask(process)
  from PhysicsTools.PatAlgos.slimming.puppiForMET_cff import makePuppiesFromMiniAOD
  makePuppiesFromMiniAOD(process,True)
  process.puppi.useExistingWeights = False
  process.puppiNoLep.useExistingWeights = False
  from PhysicsTools.PatUtils.tools.runMETCorrectionsAndUncertainties import runMetCorAndUncFromMiniAOD
  runMetCorAndUncFromMiniAOD(process,isData=False,metType="Puppi",postfix="Puppi",jetFlavor="AK4PFPuppi",recoMetFromPFCs=True,pfCandColl=cms.InputTag("puppiForMET"))
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
  # Adapt for PUPPI tune V13
  #
  process.puppi.UseFromPVLooseTight = False
  process.puppi.UseDeltaZCut = False
  process.puppi.PtMaxCharged = 20.
  process.puppi.EtaMaxCharged = 2.5
  process.puppi.PtMaxNeutralsStartSlope = 20.
  process.puppiNoLep.UseFromPVLooseTight = False
  process.puppiNoLep.UseDeltaZCut = False
  process.puppiNoLep.PtMaxCharged = 20.
  process.puppiNoLep.EtaMaxCharged = 2.5
  process.puppiNoLep.PtMaxNeutralsStartSlope = 20.
