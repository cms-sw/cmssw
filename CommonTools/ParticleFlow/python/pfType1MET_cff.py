import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.pfMET_cfi  import *
from CommonTools.ParticleFlow.pfType1MET_cfi  import *

pfRawMET = pfMET.clone(alias="pfRawMET")
#pfRawMET.hfCalibFactor = 1.
pfType1MET.inputUncorMetLabel = 'pfRawMET'

from JetMETCorrections.Configuration.JetCorrectorsAllAlgos_cff import *
pfType1MET.corrector = 'ak4PFL2L3Corrector'
pfType1METChain = cms.Sequence( ak4PFL2L3CorrectorChain * pfType1MET )

pfCorMET = cms.Sequence( pfRawMET * pfType1METChain )
