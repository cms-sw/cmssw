import FWCore.ParameterSet.Config as cms

hltHITval = cms.EDAnalyzer('ValHcalIsoTrackHLT',

produceRates=cms.bool(False), 
sampleCrossSection=cms.double(7.53E10),
numberOfEvents=cms.uint32(30000000),
luminosity=cms.double(8E29),

outputTxtFileName=cms.string("rates_minBias.txt"),

saveToRootFile=cms.bool(False),

L1FilterName=cms.string("hltL1sIsoTrack8E29"),
testL1=cms.bool(False),
doL1Prescaling=cms.bool(False),
gtDigiLabel=cms.InputTag("hltGtDigis"),
l1seedNames=cms.vstring("L1_SingleJet20U",
			"L1_SingleJet30U",
			"L1_SingleJet40U",
			"L1_SingleJet50U",
			"L1_SingleJet60U",
			"L1_SingleTauJet10U",
			"L1_SingleTauJet20U",
			"L1_SingleTauJet30U",
			"L1_SingleTauJet50U"),

hltL1extraJetLabel=cms.VInputTag("hltL1extraParticles:Tau",
                                 "hltL1extraParticles:Central",
                                 "hltL1extraParticles:Forward"),

checkL1TurnOn=cms.bool(False),
genJetsLabel=cms.InputTag("iterativeCone5GenJets"),

produceRatePdep=cms.bool(True),

folderName=cms.string("HLT/HcalIsoTrack"),
outputRootFileName=cms.string("hltHITval.root"),

debugL2=cms.bool(True),
L2producerLabelHB=cms.InputTag("hltIsolPixelTrackProdHB8E29"),
L2producerLabelHE=cms.InputTag("hltIsolPixelTrackProdHE8E29"),
L2momThresholdHB=cms.untracked.double(8),
L2momThresholdHE=cms.untracked.double(20),
L2isolationHB=cms.untracked.double(2),
L2isolationHE=cms.untracked.double(2),

hlTriggerResultsLabel=cms.InputTag("TriggerResults"),

hltProcessName=cms.string("HLT"),
hltTriggerEventLabel=cms.string("hltTriggerSummaryRAW"),

hltL3FilterLabelHB=cms.string("hltIsolPixelTrackL3FilterHB8E29"),
hltL3FilterLabelHE=cms.string("hltIsolPixelTrackL3FilterHE8E29"),
L3producerLabelHB=cms.InputTag("hltHITIPTCorrectorHB8E29"),
L3producerLabelHE=cms.InputTag("hltHITIPTCorrectorHE8E29"),
L3momThreshold=cms.untracked.double(20),


HBtriggerName=cms.string("HLT_IsoTrackHB_8E29"),
HEtriggerName=cms.string("HLT_IsoTrackHE_8E29"),

excludeFromOverlap=cms.vstring("AlCa_EcalPi0_8E29",
				"AlCa_HcalPhiSym",
				"AlCa_EcalPhiSym",
				"AlCa_EcalEta_8E29",
				"AlCa_RPCMuonNoHits",
				"AlCa_RPCMuonNormalisation",
				"HLT_StoppedHSCP_8E29"),

useReco=cms.bool(False),
recoTracksLabel=cms.InputTag("generalTracks"),

LookAtPixelTracks=cms.bool(True),
PixelTrackLabelHB=cms.InputTag("hltHITPixelTracksHB"),
PixelTrackLabelHE=cms.InputTag("hltHITPixelTracksHE")
)
