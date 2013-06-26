import FWCore.ParameterSet.Config as cms

#from HLTrigger.HLTfilters.hltHighLevel_cfi import *
#exoticaMuHLT = hltHighLevel
#Define the HLT path to be used.
#exoticaMuHLT.HLTPaths =['HLT_L1MuOpen']
#exoticaMuHLT.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT8E29")

#Define the HLT quality cut 
#exoticaHLTMuonFilter = cms.EDFilter("HLTSummaryFilter",
#    summary = cms.InputTag("hltTriggerSummaryAOD","","HLT8E29"), # trigger summary
#    member  = cms.InputTag("hltL3MuonCandidates","","HLT8E29"),      # filter or collection									
#    cut     = cms.string("pt>0"),                     # cut on trigger object
#    minN    = cms.int32(0)                  # min. # of passing objects needed
# )

Jets = cms.EDProducer("EtaPtMinCandViewSelector",
                      src = cms.InputTag("iterativeCone5CaloJets"),
                      ptMin   = cms.double(5),
                      etaMin = cms.double(-2),
                      etaMax = cms.double(2)
                      )

jetsFilter = cms.EDFilter("CandViewCountFilter",
                           src = cms.InputTag("Jets"),
                           minNumber = cms.uint32(2)
                           )

#Define the Reco quality cut
METFilter = cms.EDFilter("HLTGlobalSumsMET",
                               inputTag = cms.InputTag("htMetKT4"),
                               saveTag = cms.untracked.bool( True ),
                               observable = cms.string( "sumEt" ),
                               Min = cms.double(50.0),
                               Max = cms.double( -1.0 ),
                               MinN = cms.int32(1)
                               )
    

#Define group sequence, using HLT/Reco quality cut. 
#exoticaMuHLTQualitySeq = cms.Sequence()
METQualitySeq = cms.Sequence(
    #exoticaMuHLT+
    Jets+jetsFilter+METFilter
)

