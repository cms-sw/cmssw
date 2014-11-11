import FWCore.ParameterSet.Config as cms

from HLTriggerOffline.Higgs.hltHiggsPostProcessors_cff import *

HLTHiggsPostVal = cms.Sequence( 
		hltHiggsPostProcessors
		)

