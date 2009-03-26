import FWCore.ParameterSet.Config as cms

hltHITval = cms.EDAnalyzer('ValHcalIsoTrackHLT',

produceRates=cms.bool(False), 
sampleCrossSection=cms.double(7.53E10),
luminosity=cms.double(8E29),
outputTxtFileName=cms.string("rates_minBias.txt"),

SaveToRootFile=cms.bool(False),

doL1Prescaling=cms.bool(False),
gtDigiLabel=cms.InputTag("hltGtDigis"),
l1seedNames=cms.vstring("L1_SingleJet30",
                        "L1_SingleJet50",
                        "L1_SingleJet70",
                        "L1_SingleJet100",
                        "L1_SingleTauJet30",
                        "L1_SingleTauJet40",
                        "L1_SingleTauJet60",
                        "L1_SingleTauJet80"),

hltL1extraJetLabel=cms.VInputTag("hltL1extraParticles:Tau",
                                 "hltL1extraParticles:Central",
                                 "hltL1extraParticles:Forward"),

CheckL1TurnOn=cms.bool(True),
genJetsLabel=cms.InputTag("iterativeCone5GenJets"),

produceRatePdep=cms.bool(True),

folderName=cms.string("HLT/HcalIsoTrack"),
outputRootFileName=cms.string("hltHITval.root"),

DebugL2=cms.bool(True),
L2producerLabel=cms.InputTag("hltIsolPixelTrackProd"),

hltTriggerEventLabel=cms.InputTag("hltTriggerSummaryAOD"),
hltL3FilterLabel=cms.InputTag("hltIsolPixelTrackFilter::HLT"),

useReco=cms.bool(False),
recoTracksLabel=cms.InputTag("generalTracks")
)
