import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.pfMET_cfi  import *
from JetMETCorrections.Type1MET.pfType1MET_cfi  import *

pfRawMET = pfMET.clone(alias="pfRawMET")
#pfRawMET.hfCalibFactor = 1.
pfType1MET.inputUncorMetLabel = 'pfRawMET'

from JetMETCorrections.Configuration.JetCorrectorsAllAlgos_cff import *
pfType1MET.corrector = 'ak4PFL2L3Corrector'
pfType1METChainTask = cms.Task( ak4PFL2L3CorrectorChain , pfType1MET )
pfType1METChain = cms.Sequence( pfType1METChainTask )

pfCorMETTask = cms.Task( pfRawMET , pfType1METChainTask )
pfCorMET = cms.Sequence( pfCorMETTask )
