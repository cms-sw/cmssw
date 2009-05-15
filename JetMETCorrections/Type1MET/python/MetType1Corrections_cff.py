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
                                   corrector = cms.string('MCJetCorrectorIcone5')
                                   )

metJESCorKT4CaloJet = metJESCorIC5CaloJet.clone()
metJESCorKT4CaloJet.inputUncorJetsLabel = "kt4CaloJets"
metJESCorKT4CaloJet.corrector           = "L2L3JetCorrectorKT4Calo"

metJESCorKT6CaloJet = metJESCorIC5CaloJet.clone()
metJESCorKT6CaloJet.inputUncorJetsLabel = "kt6CaloJets"
metJESCorKT6CaloJet.corrector           = "L2L3JetCorrectorKT6Calo"

metJESCorSC5CaloJet = metJESCorIC5CaloJet.clone()
metJESCorSC5CaloJet.inputUncorJetsLabel = "sisCone5CaloJets"
metJESCorSC5CaloJet.inputUncorJetsLabel = "L2L3JetCorrectorSC5Calo"

metJESCorSC7CaloJet = metJESCorIC5CaloJet.clone()
metJESCorSC7CaloJet.inputUncorJetsLabel = "sisCone7CaloJets"
metJESCorSC7CaloJet.inputUncorJetsLabel = "L2L3JetCorrectorSC7Calo"

#MetType1Corrections = cms.Sequence(corMetType1Icone5*corMetType1Mcone5*corMetType1Mcone7)

MetType1Corrections = cms.Sequence( metJESCorIC5CaloJet*
                                    metJESCorKT4CaloJet*
                                    metJESCorKT6CaloJet*
                                    metJESCorSC5CaloJet*
                                    metJESCorSC7CaloJet)
