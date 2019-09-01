import FWCore.ParameterSet.Config as cms

from DQMOffline.JetMET.dataCertificationJetMET_cfi import *

dataCertificationJetMETSequence = cms.Sequence(qTesterJet + qTesterMET + dataCertificationJetMET)

dataCertificationJetMETSequenceHI = cms.Sequence(qTesterJet + qTesterMET + dataCertificationJetMETHI)

from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018

pp_on_AA_2018.toReplaceWith( dataCertificationJetMETSequence, dataCertificationJetMETSequenceHI )
