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
	triggerName=cms.string('HLT_IsoTrackHE'),
	l2collectionLabel=cms.string("hltIsolPixelTrackProdHE"),
	l3collectionLabel=cms.string("hltHITIPTCorrectorHE"),

	hltL3filterLabel=cms.string("hltIsolPixelTrackL3FilterHE"), 
	hltL2filterLabel=cms.string("hltIsolPixelTrackL2FilterHE"), 
	hltL1filterLabel=cms.string("hltL1sL1SingleJet52")  
	),
	cms.PSet(
        triggerName=cms.string('HLT_IsoTrackHB'),
        l2collectionLabel=cms.string("hltIsolPixelTrackProdHB"),
        l3collectionLabel=cms.string("hltHITIPTCorrectorHB"),

        hltL3filterLabel=cms.string("hltIsolPixelTrackL3FilterHB"),
        hltL2filterLabel=cms.string("hltIsolPixelTrackL2FilterHB"),
        hltL1filterLabel=cms.string("hltL1sL1SingleJet52")
        )
),
	
hltProcessName=cms.string("HLT")
)

