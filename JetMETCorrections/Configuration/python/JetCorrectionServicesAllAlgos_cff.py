import FWCore.ParameterSet.Config as cms

from JetMETCorrections.Configuration.JetCorrectionServices_cff import *


#
# SINGLE LEVEL CORRECTION SERVICES
#

# L1 (offset) Correction Services
ak8CaloL1Offset = ak4CaloL1Offset.clone()
kt4CaloL1Offset = ak4CaloL1Offset.clone()
kt6CaloL1Offset = ak4CaloL1Offset.clone()
ak4CaloL1Offset = ak4CaloL1Offset.clone()

ak8PFL1Offset   = ak4PFL1Offset.clone()
kt4PFL1Offset   = ak4PFL1Offset.clone()
kt6PFL1Offset   = ak4PFL1Offset.clone()
ak4PFL1Offset   = ak4PFL1Offset.clone()

ak8JPTL1Offset  = ak4CaloL1Offset.clone()

# L1 (fastjet) Correction Services
ak8CaloL1Fastjet = ak4CaloL1Fastjet.clone()
kt4CaloL1Fastjet = ak4CaloL1Fastjet.clone()
kt6CaloL1Fastjet = ak4CaloL1Fastjet.clone()
ak4CaloL1Fastjet = ak4CaloL1Fastjet.clone()

ak8PFL1Fastjet   = ak4PFL1Fastjet.clone()
kt4PFL1Fastjet   = ak4PFL1Fastjet.clone()
kt6PFL1Fastjet   = ak4PFL1Fastjet.clone()
ak4PFL1Fastjet   = ak4PFL1Fastjet.clone()

ak8JPTL1Fastjet  = ak4JPTL1Fastjet.clone()

# SPECIAL L1JPTOffset
ak8L1JPTOffset = ak4L1JPTOffset.clone()

# L2 (relative eta-conformity) Correction Services
ak8CaloL2Relative = ak4CaloL2Relative.clone( algorithm = 'AK8Calo' )
kt4CaloL2Relative = ak4CaloL2Relative.clone( algorithm = 'KT4Calo' )
kt6CaloL2Relative = ak4CaloL2Relative.clone( algorithm = 'KT6Calo' )
ak4CaloL2Relative = ak4CaloL2Relative.clone( algorithm = 'AK4Calo' )

ak8PFL2Relative   = ak4PFL2Relative.clone  ( algorithm = 'AK8PF' )
kt4PFL2Relative   = ak4PFL2Relative.clone  ( algorithm = 'KT4PF' )
kt6PFL2Relative   = ak4PFL2Relative.clone  ( algorithm = 'KT6PF' )
ak4PFL2Relative   = ak4PFL2Relative.clone  ( algorithm = 'AK4PF' )

# L3 (absolute) Correction Services
ak8CaloL3Absolute = ak4CaloL3Absolute.clone( algorithm = 'AK8Calo' )
kt4CaloL3Absolute = ak4CaloL3Absolute.clone( algorithm = 'KT4Calo' )
kt6CaloL3Absolute = ak4CaloL3Absolute.clone( algorithm = 'KT6Calo' )
ak4CaloL3Absolute = ak4CaloL3Absolute.clone( algorithm = 'AK4Calo' )

ak8PFL3Absolute   = ak4PFL3Absolute.clone  ( algorithm = 'AK8PF' )
kt4PFL3Absolute   = ak4PFL3Absolute.clone  ( algorithm = 'KT4PF' )
kt6PFL3Absolute   = ak4PFL3Absolute.clone  ( algorithm = 'KT6PF' )
ak4PFL3Absolute   = ak4PFL3Absolute.clone  ( algorithm = 'AK4PF' )

# Residual Correction Services
ak8CaloResidual   = ak4CaloResidual.clone()
kt4CaloResidual   = ak4CaloResidual.clone()
kt6CaloResidual   = ak4CaloResidual.clone()
ak4CaloResidual   = ak4CaloResidual.clone()

ak8PFResidual     = ak4PFResidual.clone()
kt4PFResidual     = ak4PFResidual.clone()
kt6PFResidual     = ak4PFResidual.clone()
ak4PFResidual     = ak4PFResidual.clone()

# L6 (semileptonically decaying b-jet) Correction Services
ak8CaloL6SLB = ak4CaloL6SLB.clone(
    srcBTagInfoElectron = cms.InputTag('ak8CaloJetsSoftElectronTagInfos'),
    srcBTagInfoMuon     = cms.InputTag('ak8CaloJetsSoftMuonTagInfos')
    )
kt4CaloL6SLB = ak4CaloL6SLB.clone(
    srcBTagInfoElectron = cms.InputTag('kt4CaloJetsSoftElectronTagInfos'),
    srcBTagInfoMuon     = cms.InputTag('kt4CaloJetsSoftMuonTagInfos')
    )
kt6CaloL6SLB = ak4CaloL6SLB.clone(
    srcBTagInfoElectron = cms.InputTag('kt6CaloJetsSoftElectronTagInfos'),
    srcBTagInfoMuon     = cms.InputTag('kt6CaloJetsSoftMuonTagInfos')
    )
ak4CaloL6SLB = ak4CaloL6SLB.clone(
    srcBTagInfoElectron = cms.InputTag('ak4CaloJetsSoftElectronTagInfos'),
    srcBTagInfoMuon     = cms.InputTag('ak4CaloJetsSoftMuonTagInfos')
    )

ak8PFL6SLB = ak4PFL6SLB.clone(
    srcBTagInfoElectron = cms.InputTag('ak8PFJetsSoftElectronTagInfos'),
    srcBTagInfoMuon     = cms.InputTag('ak8PFJetsSoftMuonTagInfos')
    )
kt4PFL6SLB = ak4PFL6SLB.clone(
    srcBTagInfoElectron = cms.InputTag('kt4PFJetsSoftElectronTagInfos'),
    srcBTagInfoMuon     = cms.InputTag('kt4PFJetsSoftMuonTagInfos')
    )
kt6PFL6SLB = ak4PFL6SLB.clone(
    srcBTagInfoElectron = cms.InputTag('kt6PFJetsSoftElectronTagInfos'),
    srcBTagInfoMuon     = cms.InputTag('kt6PFJetsSoftMuonTagInfos')
    )
ak4PFL6SLB = ak4PFL6SLB.clone(
    srcBTagInfoElectron = cms.InputTag('ak4PFJetsSoftElectronTagInfos'),
    srcBTagInfoMuon     = cms.InputTag('ak4PFJetsSoftMuonTagInfos')
    )


#
# MULTIPLE LEVEL CORRECTION SERVICES
#

# L2L3 CORRECTION SERVICES
ak8CaloL2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak8CaloL2Relative','ak8CaloL3Absolute')
    )
kt4CaloL2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('kt4CaloL2Relative','kt4CaloL3Absolute')
    )
kt6CaloL2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('kt6CaloL2Relative','kt6CaloL3Absolute')
    )
ak4CaloL2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak4CaloL2Relative','ak4CaloL3Absolute')
    )

ak8PFL2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak8PFL2Relative','ak8PFL3Absolute')
    )
kt4PFL2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('kt4PFL2Relative','kt4PFL3Absolute')
    )
kt6PFL2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('kt6PFL2Relative','kt6PFL3Absolute')
    )
ak4PFL2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak4PFL2Relative','ak4PFL3Absolute')
    )

#--- JPT needs the L1JPTOffset to account for the ZSP changes.
#--- L1JPTOffset is NOT the same as L1Offset !!!!!
ak8JPTL2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak8L1JPTOffset','ak8JPTL2Relative','ak8JPTL3Absolute')
    )

# L1L2L3 CORRECTION SERVICES
ak8CaloL1L2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak8CaloL1Offset','ak8CaloL2Relative','ak8CaloL3Absolute')
    )
kt4CaloL1L2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('kt4CaloL1Offset','kt4CaloL2Relative','kt4CaloL3Absolute')
    )
kt6CaloL1L2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('kt6CaloL1Offset','kt6CaloL2Relative','kt6CaloL3Absolute')
    )
ak4CaloL1L2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak4CaloL1Offset','ak4CaloL2Relative','ak4CaloL3Absolute')
    )

ak8PFL1L2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak8PFL1Offset','ak8PFL2Relative','ak8PFL3Absolute')
    )
kt4PFL1L2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('kt4PFL1Offset','kt4PFL2Relative','kt4PFL3Absolute')
    )
kt6PFL1L2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('kt6PFL1Offset','kt6PFL2Relative','kt6PFL3Absolute')
    )
ak4PFL1L2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak4PFL1Offset','ak4PFL2Relative','ak4PFL3Absolute')
    )
#--- JPT needs the L1JPTOffset to account for the ZSP changes.
#--- L1JPTOffset is NOT the same as L1Offset !!!!!
ak8JPTL1L2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak8JPTL1Offset','ak8L1JPTOffset','ak8JPTL2Relative','ak8JPTL3Absolute')
    )

# L2L3Residual CORRECTION SERVICES
ak8CaloL2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak8CaloL2Relative','ak8CaloL3Absolute','ak8CaloResidual')
    )
kt4CaloL2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('kt4CaloL2Relative','kt4CaloL3Absolute','kt4CaloResidual')
    )
kt6CaloL2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('kt6CaloL2Relative','kt6CaloL3Absolute','kt6CaloResidual')
    )
ak4CaloL2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak4CaloL2Relative','ak4CaloL3Absolute','ak4CaloResidual')
    )

ak8PFL2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak8PFL2Relative','ak8PFL3Absolute','ak8PFResidual')
    )
kt4PFL2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('kt4PFL2Relative','kt4PFL3Absolute','kt4PFResidual')
    )
kt6PFL2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('kt6PFL2Relative','kt6PFL3Absolute','kt6PFResidual')
    )
ak4PFL2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak4PFL2Relative','ak4PFL3Absolute','ak4PFResidual')
    )

# L1L2L3Residual CORRECTION SERVICES
ak8CaloL1L2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak8CaloL1Offset','ak8CaloL2Relative','ak8CaloL3Absolute','ak8CaloResidual')
    )
kt4CaloL1L2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('kt4CaloL1Offset','kt4CaloL2Relative','kt4CaloL3Absolute','kt4CaloResidual')
    )
kt6CaloL1L2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('kt6CaloL1Offset','kt6CaloL2Relative','kt6CaloL3Absolute','kt6CaloResidual')
    )
ak4CaloL1L2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak4CaloL1Offset','ak4CaloL2Relative','ak4CaloL3Absolute','ak4CaloResidual')
    )

ak8PFL1L2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak8PFL1Offset','ak8PFL2Relative','ak8PFL3Absolute','ak8PFResidual')
    )
kt4PFL1L2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('kt4PFL1Offset','kt4PFL2Relative','kt4PFL3Absolute','kt4PFResidual')
    )
kt6PFL1L2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('kt6PFL1Offset','kt6PFL2Relative','kt6PFL3Absolute','kt6PFResidual')
    )
ak4PFL1L2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak4PFL1Offset','ak4PFL2Relative','ak4PFL3Absolute','ak4PFResidual')
    )
#--- JPT needs the L1JPTOffset to account for the ZSP changes.
#--- L1JPTOffset is NOT the same as L1Offset !!!!!
ak8JPTL1L2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak8JPTL1Offset','ak8L1JPTOffset','ak8JPTL2Relative','ak8JPTL3Absolute','ak8JPTResidual')
    )

# L1FastL2L3 CORRECTION SERVICES
ak8CaloL1FastL2L3 = ak8CaloL2L3.clone()
ak8CaloL1FastL2L3.correctors.insert(0,'ak4CaloL1Fastjet')
kt4CaloL1FastL2L3 = kt4CaloL2L3.clone()
kt4CaloL1FastL2L3.correctors.insert(0,'ak4CaloL1Fastjet')
kt6CaloL1FastL2L3 = kt6CaloL2L3.clone()
kt6CaloL1FastL2L3.correctors.insert(0,'ak4CaloL1Fastjet')
ak4CaloL1FastL2L3 = ak4CaloL2L3.clone()
ak4CaloL1FastL2L3.correctors.insert(0,'ak4CaloL1Fastjet')

ak8PFL1FastL2L3 = ak8PFL2L3.clone()
ak8PFL1FastL2L3.correctors.insert(0,'ak4PFL1Fastjet')
kt4PFL1FastL2L3 = kt4PFL2L3.clone()
kt4PFL1FastL2L3.correctors.insert(0,'ak4PFL1Fastjet')
kt6PFL1FastL2L3 = kt6PFL2L3.clone()
kt6PFL1FastL2L3.correctors.insert(0,'ak4PFL1Fastjet')
ak4PFL1FastL2L3 = ak4PFL2L3.clone()
ak4PFL1FastL2L3.correctors.insert(0,'ak4PFL1Fastjet')

ak4TrackL1FastL2L3 = ak4TrackL2L3.clone()
ak4TrackL1FastL2L3.correctors.insert(0,'ak4CaloL1Fastjet')

# L1FastL2L3Residual CORRECTION SERVICES
ak8CaloL1FastL2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak8CaloL1Fastjet','ak8CaloL2Relative','ak8CaloL3Absolute','ak8CaloResidual')
    )
kt4CaloL1FastL2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('kt4CaloL1Fastjet','kt4CaloL2Relative','kt4CaloL3Absolute','kt4CaloResidual')
    )
kt6CaloL1FastL2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('kt6CaloL1Fastjet','kt6CaloL2Relative','kt6CaloL3Absolute','kt6CaloResidual')
    )
ak4CaloL1FastL2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak4CaloL1Fastjet','ak4CaloL2Relative','ak4CaloL3Absolute','ak4CaloResidual')
    )

ak8PFL1FastL2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak8PFL1Fastjet','ak8PFL2Relative','ak8PFL3Absolute','ak8PFResidual')
    )
kt4PFL1FastL2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('kt4PFL1Fastjet','kt4PFL2Relative','kt4PFL3Absolute','kt4PFResidual')
    )
kt6PFL1FastL2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('kt6PFL1Fastjet','kt6PFL2Relative','kt6PFL3Absolute','kt6PFResidual')
    )
ak4PFL1FastL2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak4PFL1Fastjet','ak4PFL2Relative','ak4PFL3Absolute','ak4PFResidual')
    )
#--- JPT needs the L1JPTOffset to account for the ZSP changes.
#--- L1JPTOffset is NOT the same as L1Offset !!!!!
ak8JPTL1FastL2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak8JPTL1Fastjet','ak8L1JPTOffset','ak8JPTL2Relative','ak8JPTL3Absolute','ak8JPTResidual')
    )

# L2L3L6 CORRECTION SERVICES
ak8CaloL2L3L6 = ak8CaloL2L3.clone()
ak8CaloL2L3L6.correctors.append('ak8CaloL6SLB')
kt4CaloL2L3L6 = kt4CaloL2L3.clone()
kt4CaloL2L3L6.correctors.append('kt4CaloL6SLB')
kt6CaloL2L3L6 = kt6CaloL2L3.clone()
kt6CaloL2L3L6.correctors.append('kt6CaloL6SLB')
ak4CaloL2L3L6 = ak4CaloL2L3.clone()
ak4CaloL2L3L6.correctors.append('ak4CaloL6SLB')

ak8PFL2L3L6 = ak8PFL2L3.clone()
ak8PFL2L3L6.correctors.append('ak8PFL6SLB')
kt4PFL2L3L6 = kt4PFL2L3.clone()
kt4PFL2L3L6.correctors.append('kt4PFL6SLB')
kt6PFL2L3L6 = kt6PFL2L3.clone()
kt6PFL2L3L6.correctors.append('kt6PFL6SLB')
ak4PFL2L3L6 = ak4PFL2L3.clone()
ak4PFL2L3L6.correctors.append('ak4PFL6SLB')


# L1L2L3L6 CORRECTION SERVICES
ak8CaloL1FastL2L3L6 = ak8CaloL1L2L3.clone()
ak8CaloL1FastL2L3L6.correctors.append('ak8CaloL6SLB')
kt4CaloL1FastL2L3L6 = kt4CaloL1L2L3.clone()
kt4CaloL1FastL2L3L6.correctors.append('kt4CaloL6SLB')
kt6CaloL1FastL2L3L6 = kt6CaloL1L2L3.clone()
kt6CaloL1FastL2L3L6.correctors.append('kt6CaloL6SLB')
ak4CaloL1FastL2L3L6 = ak4CaloL1L2L3.clone()
ak4CaloL1FastL2L3L6.correctors.append('ak4CaloL6SLB')

ak8PFL1FastL2L3L6 = ak8PFL1FastL2L3.clone()
ak8PFL1FastL2L3L6.correctors.append('ak8PFL6SLB')
kt4PFL1FastL2L3L6 = kt4PFL1FastL2L3.clone()
kt4PFL1FastL2L3L6.correctors.append('kt4PFL6SLB')
kt6PFL1FastL2L3L6 = kt6PFL1FastL2L3.clone()
kt6PFL1FastL2L3L6.correctors.append('kt6PFL6SLB')
ak4PFL1FastL2L3L6 = ak4PFL1FastL2L3.clone()
ak4PFL1FastL2L3L6.correctors.append('ak4PFL6SLB')
