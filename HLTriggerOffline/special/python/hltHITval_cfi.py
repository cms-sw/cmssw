import FWCore.ParameterSet.Config as cms

hltHITval = cms.EDAnalyzer('DQMHcalIsoTrackAlCaRaw',

produceRates=cms.bool(False), 
sampleCrossSection=cms.double(7.53E10),
luminosity=cms.double(1E31),
outputTxtFileName=cms.string("rates_minBias.txt"),
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
CheckL1TurnOn=cms.bool(False),
genJetsLabel=cms.InputTag("iterativeCone5GenJets"),

folderName=cms.string("HLT/AlCa_IsoTrack"),
outputRootFileName=cms.string("hltHITval.root"),

DebugL2=cms.bool(False),
L2producerLabel=cms.InputTag("hltIsolPixelTrackProd"),

hltTriggerEventLabel=cms.InputTag("hltTriggerSummaryAOD"),
hltL3FilterLabel=cms.InputTag("hltIsolPixelTrackFilter::HLT5"),

useReco=cms.bool(False),
recoTracksLabel=cms.InputTag("generalTracks")
)
