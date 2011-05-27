import FWCore.ParameterSet.Config as cms


import CondCore.DBCommon.CondDBSetup_cfi
import TrackingTools.TransientTrack.TransientTrackBuilder_cfi

import L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff

#import HLTrigger.HLTfilters.hltLevel1GTSeed_cfi as trig
#TauSkimTrigger = trig.hltLevel1GTSeed.clone()
#TauSkimTrigger.L1TechTriggerSeeding = cms.bool(True)
#TauSkimTrigger.L1SeedsLogicalExpression = cms.string('(0 AND (40 OR 41) AND NOT (36 OR 37 OR 38 OR 39))')

#TauSkimScraping = cms.EDFilter("FilterOutScraping",
#   	applyfilter = cms.untracked.bool(True),
#   	debugOn = cms.untracked.bool(False),
#       numtrack = cms.untracked.uint32(10),
#       thresh = cms.untracked.double(0.25)
#)

TauSkimPFTausSelected = cms.EDFilter("PFTauSelector",
   src = cms.InputTag("hpsPFTauProducer"),
   discriminators = cms.VPSet(
	cms.PSet( discriminator=cms.InputTag("hpsPFTauDiscriminationByDecayModeFinding"),
		   selectionCut=cms.double(0.5)           
	),
   cms.PSet( discriminator=cms.InputTag("hpsPFTauDiscriminationByLooseIsolation"),
		   selectionCut=cms.double(0.5)           
	),

   ),
   cut = cms.string('et > 15. && abs(eta) < 2.5') 
)

TauSkimPFTauSkimmed = cms.EDFilter("CandViewCountFilter",
 src = cms.InputTag('TauSkimPFTausSelected'),
 minNumber = cms.uint32(2)
)

tauSkimSequence = cms.Sequence(
#   TauSkimTrigger *
#   TauSkimScraping *
   TauSkimPFTausSelected *
   TauSkimPFTauSkimmed
   )

