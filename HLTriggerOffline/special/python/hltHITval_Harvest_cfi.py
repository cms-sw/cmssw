import FWCore.ParameterSet.Config as cms

hltHITval_Harvest = cms.EDFilter("PostProcessor",
    outputFileName = cms.untracked.string(''),
    verbose        = cms.untracked.uint32(0),
    commands       = cms.vstring(),
    resolution     = cms.vstring(),                                    
    subDirs         = cms.untracked.vstring('HLT/HcalIsoTrack/'),
    efficiency     = cms.vstring(
        "TurnOnLead 'L1 seeding turn on for leading jet; gen jet pT (GeV); N_L1/N_All' hpTgenLeadL1 hpTgenLead",
	"TurnOnNextToLead 'L1 seeding turn on for next-to-leading jet; gen jet pT (GeV); N_L1/N_All' hpTgenNextL1 hpTgenNext"
    )
)


