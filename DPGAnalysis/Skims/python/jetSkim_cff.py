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

Jet2 = cms.EDFilter("EtaPtMinCandViewSelector",
                      src = cms.InputTag("iterativeCone5CaloJets"),
                      ptMin   = cms.double(8),
                      etaMin = cms.double(-2),
                      etaMax = cms.double(2)
                      )

Jet1 = cms.EDFilter("EtaPtMinCandViewSelector",
                      src = cms.InputTag("Jet2"),
                      ptMin   = cms.double(8),
                      etaMin = cms.double(-1),
                      etaMax = cms.double(1)
                      )
#Define the Reco quality cut
#jetFilter = cms.EDFilter("CaloJetSelector",
#                               src = cms.InputTag("iterativeCone5CaloJets"),
#                               cut = cms.string('pt > 100 && abs(eta) < 2.0' ),
#                               filter = cms.bool(True),
#                              minNumber = cms.uint32(2)
#                                sizeSelector = cms.uint32(2)
#                               )
                               
dijetFilter = cms.EDFilter("CandViewCountFilter",
                           src = cms.InputTag("Jet2"),
                           minNumber = cms.uint32(2)
                           )

jetFilter = cms.EDFilter("CandViewCountFilter",
                               src = cms.InputTag("Jet1"),
                               minNumber = cms.uint32(1)
                               )

#===== add electrons =======

jetSuperClusterMerger =  cms.EDFilter("EgammaSuperClusterMerger",
                                   src = cms.VInputTag(cms.InputTag('correctedHybridSuperClusters'),
                                                       cms.InputTag('correctedMulti5x5SuperClustersWithPreshower'))
                                   )
jetSuperClusterCands = cms.EDProducer("ConcreteEcalCandidateProducer",
                                   src = cms.InputTag("jetSuperClusterMerger"),
                                   particleType = cms.string('e-')
                                   )

goodJetSuperClusters = cms.EDFilter("CandViewRefSelector",
                                 src = cms.InputTag("jetSuperClusterCands"),
                                 cut = cms.string('et > 3.0')
                                 )

jetSuperClusterPt5Filter = cms.EDFilter("CandViewCountFilter",
                                      src = cms.InputTag("goodJetSuperClusters"),
                                      minNumber = cms.uint32(2)
                                      )

twoEmClusters = cms.Sequence(
    jetSuperClusterMerger+jetSuperClusterCands+goodJetSuperClusters+jetSuperClusterPt5Filter
    )

#Define group sequence, using HLT/Reco quality cut. 
#exoticaMuHLTQualitySeq = cms.Sequence()
jetRecoQualitySeq = cms.Sequence(
#    twoEmClusters +
    Jet2+Jet1+dijetFilter+jetFilter
)

