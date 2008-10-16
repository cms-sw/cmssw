import FWCore.ParameterSet.Config as cms

hltHITval = cms.EDAnalyzer('DQMHcalIsoTrackAlCaRaw',
sampleCrossSection=cms.double(3.658E7),
luminosity=cms.double(2E31),
outputTxtFileName=cms.string("rates.txt"),
folderName=cms.string("HLT/AlCa_IsoTrack"),
outputRootFileName=cms.string("hltHITval.root"),
hltTriggerEventLabel=cms.InputTag("hltTriggerSummaryAOD"),
hltL3FilterLabel=cms.InputTag("hltIsolPixelTrackFilter::HLT"),
hltL1extraJetLabel=cms.VInputTag("hltL1extraParticles:Tau",
				 "hltL1extraParticles:Central",
				 "hltL1extraParticles:Forward"),
l1seedNames=cms.vstring("L1_SingleJet30",
			"L1_SingleJet50",
			"L1_SingleJet70",
			"L1_SingleJet100",
			"L1_SingleTauJet30",
			"L1_SingleTauJet40",
			"L1_SingleTauJet60",
			"L1_SingleTauJet80"),
gtDigiLabel=cms.InputTag("hltGtDigis"),
useReco=cms.bool(True),
recoTracksLabel=cms.InputTag("generalTracks")

)
