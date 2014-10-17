import FWCore.ParameterSet.Config as cms

from JetMETCorrections.Configuration.JetCorrectors_cff import *

# L1 Correction Producers
ak4CaloJetsL1 = cms.EDProducer(
    'CaloJetCorrectionProducer',
    src        = cms.InputTag('ak4CaloJets'),
    correctors = cms.VInputTag('L1FastjetCorrector')
    )
ak4PFJetsL1 = cms.EDProducer(
    'PFJetCorrectionProducer',
    src        = cms.InputTag('ak4PFJets'),
    correctors = cms.VInputTag('L1FastjetCorrector')
    )
ak4PFCHSJetsL1 = cms.EDProducer(
    'PFJetCorrectionProducer',
    src        = cms.InputTag('ak4PFJetsCHS'),
    correctors = cms.VInputTag('L1FastjetCorrector')
    )

ak4JPTJetsL1 = cms.EDProducer(
    'JPTJetCorrectionProducer',
    src        = cms.InputTag('JetPlusTrackZSPCorJetAntiKt4'),
    correctors = cms.VInputTag('L1FastjetCorrector')
    )
ak4TrackJetsL1 = cms.EDProducer(
    'TrackJetCorrectionProducer',
    src        = cms.InputTag('ak4TrackJets'),
    correctors = cms.VInputTag('L1FastjetCorrector')
    )



# L2L3 Correction Producers
ak4CaloJetsL2 = ak4CaloJetsL1.clone(correctors = ['ak4CaloL2RelativeCorrector'])
ak4PFJetsL2 = ak4PFJetsL1.clone(correctors = ['ak4PFL2RelativeCorrector'])
ak4PFCHSJetsL2 = ak4PFCHSJetsL1.clone(correctors = ['ak4PFCHSL2RelativeCorrector'])
ak4JPTJetsL2 = ak4JPTJetsL1.clone(correctors = ['ak4JPTL2RelativeCorrector'])
ak4TrackJetsL2 = ak4TrackJetsL1.clone(correctors = ['ak5TRKL2RelativeCorrector'])

# L2L3 Correction Producers
ak4CaloJetsL2L3 = ak4CaloJetsL1.clone(correctors = ['ak4CaloL2L3Corrector'])
ak4PFJetsL2L3 = ak4PFJetsL1.clone(correctors = ['ak4PFL2L3Corrector'])
ak4PFCHSJetsL2L3 = ak4PFCHSJetsL1.clone(correctors = ['ak4PFCHSL2L3Corrector'])
ak4JPTJetsL2L3 = ak4JPTJetsL1.clone(correctors = ['ak4JPTL2L3Corrector'])
ak4TrackJetsL2L3 = ak4TrackJetsL1.clone(correctors = ['ak5TRKL2L3Corrector'])


# L1L2L3 Correction Producers
ak4CaloJetsL1L2L3 = ak4CaloJetsL1.clone(correctors = ['ak4CaloL1L2L3Corrector'])
ak4PFJetsL1L2L3 = ak4PFJetsL1.clone(correctors = ['ak4PFL1L2L3Corrector'])
ak4PFCHSJetsL1L2L3 = ak4PFCHSJetsL1.clone(correctors = ['ak4PFCHSL1L2L3Corrector'])
ak4JPTJetsL1L2L3 = ak4JPTJetsL1.clone(correctors = ['ak4JPTL1L2L3Corrector'])
ak4TrackJetsL1L2L3 = ak4TrackJetsL1.clone(correctors = ['ak5TRKL1L2L3Corrector'])

# L2L3L6 CORRECTION PRODUCERS
ak4CaloJetsL2L3L6 = ak4CaloJetsL1.clone(correctors = ['ak4CaloL2L3L6Corrector'])
ak4PFJetsL2L3L6 = ak4PFJetsL1.clone(correctors = ['ak4PFL2L3L6Corrector'])


# L1L2L3L6 CORRECTION PRODUCERS
ak4CaloJetsL1L2L3L6 = ak4CaloJetsL1.clone(correctors = ['ak4CaloL1L2L3L6Corrector'])
ak4PFJetsL1L2L3L6 = ak4PFJetsL1.clone(correctors = ['ak4PFL1L2L3L6Corrector'])

