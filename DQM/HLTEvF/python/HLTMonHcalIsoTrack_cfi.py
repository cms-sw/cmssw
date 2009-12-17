import FWCore.ParameterSet.Config as cms

hltMonHcalIsoTrack = cms.EDAnalyzer('HLTMonHcalIsoTrack',
folderName=cms.string("HLT/HCAL/HLT_IsoTrack"),
SaveToRootFile=cms.bool(False),
outputRootFileName=cms.string("hltHITdqm.root"),
useProducerCollections=cms.bool(True),
hltRAWTriggerEventLabel=cms.string("hltTriggerSummaryRAW"),
hltAODTriggerEventLabel=cms.string("hltTriggerSummaryAOD"),

triggers=cms.VPSet(
	cms.PSet(
	triggerName=cms.string('HLT_IsoTrackHE_8E29'),
	l2collectionLabel=cms.string("hltIsolPixelTrackProdHE8E29"),
	l3collectionLabel=cms.string("hltHITIPTCorrectorHE8E29"),

	hltL3filterLabel=cms.string("hltIsolPixelTrackL3FilterHE8E29"), 
	hltL2filterLabel=cms.string("hltIsolPixelTrackL2FilterHE8E29"), 
	hltL1filterLabel=cms.string("hltL1sIsoTrack8E29")  
	),
	cms.PSet(
        triggerName=cms.string('HLT_IsoTrackHB_8E29'),
        l2collectionLabel=cms.string("hltIsolPixelTrackProdHB8E29"),
        l3collectionLabel=cms.string("hltHITIPTCorrectorHB8E29"),

        hltL3filterLabel=cms.string("hltIsolPixelTrackL3FilterHB8E29"),
        hltL2filterLabel=cms.string("hltIsolPixelTrackL2FilterHB8E29"),
        hltL1filterLabel=cms.string("hltL1sIsoTrack8E29")
        )
),
	
hltProcessName=cms.string("HLT")
)

