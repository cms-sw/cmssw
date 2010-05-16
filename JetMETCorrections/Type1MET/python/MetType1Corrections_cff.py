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
                                   corrector = cms.string('L2L3JetCorrectorIC5Calo'),
                                   UscaleA = cms.double(1.2),
                                   UscaleB = cms.double(2.1),
                                   UscaleC = cms.double(0.6),
                                   useTypeII = cms.bool(False),
                                   hasMuonsCorr = cms.bool(False)
                                   )

metJESCorKT4CaloJet = metJESCorIC5CaloJet.clone()
metJESCorKT4CaloJet.inputUncorJetsLabel = "kt4CaloJets"
metJESCorKT4CaloJet.corrector           = "L2L3JetCorrectorKT4Calo"

metJESCorKT6CaloJet = metJESCorIC5CaloJet.clone()
metJESCorKT6CaloJet.inputUncorJetsLabel = "kt6CaloJets"
metJESCorKT6CaloJet.corrector           = "L2L3JetCorrectorKT6Calo"

metJESCorAK5CaloJet = metJESCorIC5CaloJet.clone()
metJESCorAK5CaloJet.inputUncorJetsLabel = "ak5CaloJets"
metJESCorAK5CaloJet.corrector           = "L2L3JetCorrectorAK5Calo"

metJESCorAK7CaloJet = metJESCorIC5CaloJet.clone()
metJESCorAK7CaloJet.inputUncorJetsLabel = "ak7CaloJets"
metJESCorAK7CaloJet.corrector           = "L2L3JetCorrectorAK7Calo"

metJESCorSC5CaloJet = metJESCorIC5CaloJet.clone()
metJESCorSC5CaloJet.inputUncorJetsLabel = "sisCone5CaloJets"
metJESCorSC5CaloJet.corrector = "L2L3JetCorrectorSC5Calo"

metJESCorSC7CaloJet = metJESCorIC5CaloJet.clone()
metJESCorSC7CaloJet.inputUncorJetsLabel = "sisCone7CaloJets"
metJESCorSC7CaloJet.corrector = "L2L3JetCorrectorSC7Calo"

#MetType1Corrections = cms.Sequence(corMetType1Icone5*corMetType1Mcone5*corMetType1Mcone7)

MetType1Corrections = cms.Sequence( metJESCorIC5CaloJet*
                                    metJESCorKT4CaloJet*
                                    metJESCorKT6CaloJet*
                                    metJESCorAK5CaloJet*
                                    metJESCorAK7CaloJet*
                                    metJESCorSC5CaloJet*
                                    metJESCorSC7CaloJet)
