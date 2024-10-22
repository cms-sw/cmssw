import FWCore.ParameterSet.Config as cms


# run on MIONAOD
RUN_ON_MINIAOD = False


# cuts
JET_CUT=("pt > 1000 && abs(eta)<5.0")

# single lepton selectors
if RUN_ON_MINIAOD:
    teVJets = cms.EDFilter("CandViewRefSelector",
                           src = cms.InputTag("slimmedJets"),
                           cut = cms.string(JET_CUT)
    )
else:
    teVJets = cms.EDFilter("CandViewRefSelector",
                           src = cms.InputTag("ak4PFJets"),
                           cut = cms.string(JET_CUT)
                       )
    
teVJetsCountFilter = cms.EDFilter("CandViewCountFilter",
                                  src = cms.InputTag("teVJets"),
                                  minNumber = cms.uint32(1)
                              )
                           


#sequences
teVJetSequence = cms.Sequence(teVJets*teVJetsCountFilter )
