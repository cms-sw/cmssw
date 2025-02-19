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
                               

#Define the Reco quality cut
singleRecoMuonPt20Filter = cms.EDFilter("MuonRefSelector",
                                        src = cms.InputTag("muons"),
                                        cut = cms.string('pt > 20.0 && abs(eta) < 2.1' ),
                                        filter = cms.bool(True),
                                        minN    = cms.int32(1) 
                                        )

singleRecoMuonPt15Filter = cms.EDFilter("MuonRefSelector",
                                        src = cms.InputTag("muons"),
                                        cut = cms.string('pt > 15.0 && abs(eta) < 2.1' ),
                                        filter = cms.bool(True),
                                        minN    = cms.int32(1)
                                        )

singleRecoMuonPt10Filter = cms.EDFilter("MuonRefSelector",
                                        src = cms.InputTag("muons"),
                                        cut = cms.string('pt > 10.0 && abs(eta) < 2.1' ),
                                        filter = cms.bool(True),
                                        minN    = cms.int32(1)
                                        )

singleRecoMuonPt5Filter = cms.EDFilter("MuonRefSelector",
                                        src = cms.InputTag("muons"),
                                        cut = cms.string('pt > 0.5 && abs(eta) < 2.1 && isGlobalMuon == 1  && abs (track().d0) < 2 ' ),
                                        filter = cms.bool(True),
                                        minN    = cms.int32(1)
                                        )


muonSCMerger =  cms.EDFilter("EgammaSuperClusterMerger",
                                   src = cms.VInputTag(cms.InputTag('correctedHybridSuperClusters'),
                                                       cms.InputTag('correctedMulti5x5SuperClustersWithPreshower'))
                                   )
muonSCCands = cms.EDProducer("ConcreteEcalCandidateProducer",
                                   src = cms.InputTag("muonSCMerger"),
                                   particleType = cms.string('mu+')
                                   )

goodMuonSCs = cms.EDFilter("CandViewRefSelector",
                                 src = cms.InputTag("muonSCCands"),
                                 cut = cms.string('et > 2.0')
                                 )

muonSCPt5Filter = cms.EDFilter("CandViewCountFilter",
                                     src = cms.InputTag("goodMuonSCs"),
                                     minNumber = cms.uint32(2)
                                     )

muonEmClusters = cms.Sequence(
    muonSCMerger+muonSCCands+goodMuonSCs+muonSCPt5Filter
    )


#Define group sequence, using HLT/Reco quality cut. 
#singleMuHLTQualitySeq = cms.Sequence()
singleMuPt20RecoQualitySeq = cms.Sequence(
    #singleMuHLT+
    singleRecoMuonPt20Filter
)

singleMuPt15RecoQualitySeq = cms.Sequence(
    #singleMuHLT+
    singleRecoMuonPt15Filter
)

singleMuPt10RecoQualitySeq = cms.Sequence(
    #singleMuHLT+
    singleRecoMuonPt10Filter
)

singleMuPt5RecoQualitySeq = cms.Sequence(
        #singleMuHLT+
#    muonEmClusters+
        singleRecoMuonPt5Filter
        )
