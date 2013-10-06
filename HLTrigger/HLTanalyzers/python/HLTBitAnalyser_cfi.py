import FWCore.ParameterSet.Config as cms

hltbitanalysis = cms.EDAnalyzer("HLTBitAnalyzer",
    ### Trigger objects
    l1GctHFBitCounts                = cms.InputTag("hltGctDigis"),
    l1GctHFRingSums                 = cms.InputTag("hltGctDigis"),
    l1GtObjectMapRecord             = cms.InputTag("hltL1GtObjectMap::HLT"),
    l1GtReadoutRecord               = cms.InputTag("hltGtDigis::HLT"),

    l1extramc                       = cms.string('hltL1extraParticles'),
    l1extramu                       = cms.string('hltL1extraParticles'),
    hltresults                      = cms.InputTag("TriggerResults::HLT"),
    HLTProcessName                  = cms.string("HLT"),

    ### GEN objects
    mctruth                         = cms.InputTag("genParticles::SIM"),
    genEventInfo                    = cms.InputTag("generator::SIM"),

    ### SIM objects
    simhits                         = cms.InputTag("g4SimHits"),
                                
    ## reco vertices
    OfflinePrimaryVertices0     = cms.InputTag('offlinePrimaryVertices'),
                                
    ### Run parameters
    RunParameters = cms.PSet(
    HistogramFile = cms.untracked.string('hltbitanalysis.root'),
    Monte         = cms.bool(True)
                                    
    )
                                
)
