import FWCore.ParameterSet.Config as cms

from HLTriggerOffline.Exotica.hltExoticaPostProcessors_cff import *

HLTExoticaPostVal = cms.Sequence( 
		hltExoticaPostProcessors
		)

