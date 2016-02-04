import FWCore.ParameterSet.Config as cms

# File: MetCorrections.cff
# Author: R. Cavanaugh
# Date: 08.08.2006
#
# Updated:  Added modules for MET corrections with KT, Siscone jet algorithms

metJESCorIC5CaloJet = cms.EDProducer("Type1MET",
                                   inputUncorJetsLabel = cms.string('iterativeCone5CaloJets'),
                                   jetEMfracLimit = cms.double(0.9),
                                   metType = cms.string('CaloMET'),
                                   jetPTthreshold = cms.double(20.0),
                                   inputUncorMetLabel = cms.string('met'),
                                   UscaleA = cms.double(1.5),
                                   UscaleB = cms.double(1.8),
                                   UscaleC = cms.double(-0.06),
                                   useTypeII = cms.bool(False),
                                   hasMuonsCorr = cms.bool(False),
                                   corrector = cms.string('ic5CaloL2L3')
                                   )

metJESCorAK5PFJet = cms.EDProducer("Type1MET",
                                   inputUncorJetsLabel = cms.string('ak5PFJets'),
                                   jetEMfracLimit = cms.double(0.9),
                                   metType = cms.string('PFMET'),
                                   jetPTthreshold = cms.double(1.0),
                                   inputUncorMetLabel = cms.string('pfMet'),
                                   UscaleA = cms.double(1.5),
                                   UscaleB = cms.double(1.8),
                                   UscaleC = cms.double(-0.06),
                                   useTypeII = cms.bool(False),
                                   hasMuonsCorr = cms.bool(False),
                                   corrector = cms.string('ak5PFL2L3')
                                   )


metJESCorKT4CaloJet = metJESCorIC5CaloJet.clone()
metJESCorKT4CaloJet.inputUncorJetsLabel = "kt4CaloJets"
metJESCorKT4CaloJet.corrector           = "kt4CaloL2L3"

metJESCorKT6CaloJet = metJESCorIC5CaloJet.clone()
metJESCorKT6CaloJet.inputUncorJetsLabel = "kt6CaloJets"
metJESCorKT6CaloJet.corrector           = "kt6CaloL2L3"

metJESCorAK5CaloJet = metJESCorIC5CaloJet.clone()
metJESCorAK5CaloJet.inputUncorJetsLabel = "ak5CaloJets"
metJESCorAK5CaloJet.corrector           = "ak5CaloL2L3"

metJESCorAK7CaloJet = metJESCorIC5CaloJet.clone()
metJESCorAK7CaloJet.inputUncorJetsLabel = "ak7CaloJets"
metJESCorAK7CaloJet.corrector           = "ak7CaloL2L3"

metJESCorSC5CaloJet = metJESCorIC5CaloJet.clone()
metJESCorSC5CaloJet.inputUncorJetsLabel = "sisCone5CaloJets"
metJESCorSC5CaloJet.corrector = "sisCone5CaloL2L3"

metJESCorSC7CaloJet = metJESCorIC5CaloJet.clone()
metJESCorSC7CaloJet.inputUncorJetsLabel = "sisCone7CaloJets"
metJESCorSC7CaloJet.corrector = "sisCone7CaloL2L3"

#MetType1Corrections = cms.Sequence(corMetType1Icone5*corMetType1Mcone5*corMetType1Mcone7)

MetType1Corrections = cms.Sequence( metJESCorIC5CaloJet*
                                    metJESCorKT4CaloJet*
                                    metJESCorKT6CaloJet*
                                    metJESCorAK5CaloJet*
                                    metJESCorAK7CaloJet*
                                    metJESCorSC5CaloJet*
                                    metJESCorSC7CaloJet)
