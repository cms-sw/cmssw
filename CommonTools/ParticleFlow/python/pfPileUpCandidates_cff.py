import FWCore.ParameterSet.Config as cms

#Create the PU candidates
pfPileUpCandidates = cms.EDProducer("TPPFCandidatesOnPFCandidates",
                                    enable =  cms.bool( True ),
                                    verbose = cms.untracked.bool( False ),
                                    name = cms.untracked.string("pileUpCandidates"),
                                    topCollection = cms.InputTag("pfNoPileUp"),
                                    bottomCollection = cms.InputTag("particleFlowTmp")
                                    )


#Take the PU charged particles
pfPUChargedCandidates = cms.EDFilter("PdgIdPFCandidateSelector",
                                     src = cms.InputTag("pfPileUpCandidates"),
                                     pdgId = cms.vint32(211,-211,321,-321,999211,2212,-2212,11,-11,13,-13)
                                     )


#Create All Charged Particles
pfAllChargedCandidates = cms.EDFilter("PdgIdPFCandidateSelector",
                                      src = cms.InputTag("pfNoPileUp"),
                                      pdgId = cms.vint32(211,-211,321,-321,999211,2212,-2212,11,-11,13,-13)
                                      )


pfPileUpCandidatesSequence = cms.Sequence(pfPileUpCandidates+
                                          pfPUChargedCandidates+
                                          pfAllChargedCandidates)

