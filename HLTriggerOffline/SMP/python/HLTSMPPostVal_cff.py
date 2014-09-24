import FWCore.ParameterSet.Config as cms

from HLTriggerOffline.SMP.hltSMPPostProcessors_cff import *

HLTSMPPostVal = cms.Sequence( 
		hltSMPPostProcessors
		)

