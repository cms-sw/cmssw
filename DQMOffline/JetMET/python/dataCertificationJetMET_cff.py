import FWCore.ParameterSet.Config as cms

from DQMOffline.JetMET.dataCertificationJetMET_cfi import *

dataCertificationJetMETSequence = cms.Sequence(qTesterJet + qTesterMET + dataCertificationJetMET)

dataCertificationJetMETSequenceHI = cms.Sequence(qTesterJet + qTesterMET + dataCertificationJetMETHI)

from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA

pp_on_AA.toReplaceWith( dataCertificationJetMETSequence, dataCertificationJetMETSequenceHI )
