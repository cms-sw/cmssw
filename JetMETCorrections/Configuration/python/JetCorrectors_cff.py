import FWCore.ParameterSet.Config as cms

from JetMETCorrections.Configuration.JetCorrectionServices_cff import *

#
# SINGLE LEVEL CORRECTORS
#

# L1 (Offset) CORRECTOR
ak4CaloL1OffsetCorrector = cms.EDProducer(
    'L1OffsetCorrectorProducer',
    level = cms.string('L1Offset'),
    algorithm = cms.string('AK4Calo'),
    vertexCollection = cms.string('offlinePrimaryVertices'),
    minVtxNdof = cms.int32(4)
    )

ak4PFL1OffsetCorrector = ak4CaloL1OffsetCorrector.clone(algorithm = 'AK4PF')
ak4PFCHSL1OffsetCorrector = ak4CaloL1OffsetCorrector.clone(algorithm = 'AK4PFchs')
ak4JPTL1OffsetCorrector = ak4CaloL1OffsetCorrector.clone(algorithm = 'AK4JPT')

# L1 (JPT Offset) CORRECTOR
ak4L1JPTOffsetCorrector = cms.EDProducer(
    'L1JPTOffsetCorrectorProducer',
    level = cms.string('L1JPTOffset'),
    algorithm = cms.string('AK4JPT'),
    offsetService = cms.string('ak4CaloL1Offset')
    )

# L1 (Fastjet PU Subtraction) CORRECTOR
ak4CaloL1FastjetCorrector = cms.EDProducer(
    'L1FastjetCorrectorProducer',
    level       = cms.string('L1FastJet'),
    algorithm   = cms.string('AK4Calo'),
    srcRho      = cms.InputTag( 'fixedGridRhoFastjetAllCalo'  )
    )
ak4PFL1FastjetCorrector = cms.EDProducer(
    'L1FastjetCorrectorProducer',
    level       = cms.string('L1FastJet'),
    algorithm   = cms.string('AK4PF'),
    srcRho      = cms.InputTag( 'fixedGridRhoFastjetAll' )
    )
ak4PFCHSL1FastjetCorrector = cms.EDProducer(
    'L1FastjetCorrectorProducer',
    level       = cms.string('L1FastJet'),
    algorithm   = cms.string('AK4PFchs'),
    srcRho      = cms.InputTag( 'fixedGridRhoFastjetAll' )
    )
ak4JPTL1FastjetCorrector = ak4CaloL1FastjetCorrector.clone()

# L2 (relative eta-conformity) CORRECTORs
ak4CaloL2RelativeCorrector = cms.EDProducer(
    'LXXXCorrectorProducer',
    level     = cms.string('L2Relative'),
    algorithm = cms.string('AK4Calo')
    )
ak4PFL2RelativeCorrector = ak4CaloL2RelativeCorrector.clone( algorithm = 'AK4PF' )
ak4PFCHSL2RelativeCorrector = ak4CaloL2RelativeCorrector.clone( algorithm = 'AK4PFchs' )
ak4JPTL2RelativeCorrector = ak4CaloL2RelativeCorrector.clone( algorithm = 'AK4JPT' )
ak4TrackL2RelativeCorrector = ak4CaloL2RelativeCorrector.clone( algorithm = 'AK4TRK' )

# L3 (absolute) CORRECTORs
ak4CaloL3AbsoluteCorrector = cms.EDProducer(
    'LXXXCorrectorProducer',
    level     = cms.string('L3Absolute'),
    algorithm = cms.string('AK4Calo')
    )
ak4PFL3AbsoluteCorrector     = ak4CaloL3AbsoluteCorrector.clone( algorithm = 'AK4PF' )
ak4PFCHSL3AbsoluteCorrector     = ak4CaloL3AbsoluteCorrector.clone( algorithm = 'AK4PFchs' )
ak4JPTL3AbsoluteCorrector    = ak4CaloL3AbsoluteCorrector.clone( algorithm = 'AK4JPT' )
ak4TrackL3AbsoluteCorrector  = ak4CaloL3AbsoluteCorrector.clone( algorithm = 'AK4TRK' )

# Residual CORRECTORs
ak4CaloResidualCorrector = cms.EDProducer(
    'LXXXCorrectorProducer',
    level     = cms.string('L2L3Residual'),
    algorithm = cms.string('AK4Calo')
    )
ak4PFResidualCorrector  = ak4CaloResidualCorrector.clone( algorithm = 'AK4PF' )
ak4PFCHSResidualCorrector  = ak4CaloResidualCorrector.clone( algorithm = 'AK4PFchs' )
ak4JPTResidualCorrector = ak4CaloResidualCorrector.clone( algorithm = 'AK4JPT' )

# L6 (semileptonically decaying b-jet) Correction Services
ak4CaloL6SLBCorrector = cms.EDProducer(
    'L6SLBCorrectorProduce',
    level               = cms.string('L6SLB'),
    algorithm           = cms.string(''),
    addMuonToJet        = cms.bool(True),
    srcBTagInfoElectron = cms.InputTag('ak4CaloJetsSoftElectronTagInfos'),
    srcBTagInfoMuon     = cms.InputTag('ak4CaloJetsSoftMuonTagInfos')
    )
ak4PFL6SLBCorrector = cms.EDProducer(
    'L6SLBCorrectorProduce',
    level               = cms.string('L6SLB'),
    algorithm           = cms.string(''),
    addMuonToJet        = cms.bool(False),
    srcBTagInfoElectron = cms.InputTag('ak4PFJetsSoftElectronTagInfos'),
    srcBTagInfoMuon     = cms.InputTag('ak4PFJetsSoftMuonTagInfos')
    )


#
# MULTIPLE LEVEL CORRECTORS
#

# L2L3 CORRECORNS
ak4CaloL2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4CaloL2Relative','ak4CaloL3Absolute')
    )
ak4PFL2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4PFL2Relative','ak4PFL3Absolute')
    )
ak4PFCHSL2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4PFCHSL2Relative','ak4PFCHSL3Absolute')
    )
#--- JPT needs the L1JPTOffset to account for the ZSP changes.
#--- L1JPTOffset is NOT the same as L1Offset !!!!!
ak4JPTL2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4L1JPTOffset','ak4JPTL2Relative','ak4JPTL3Absolute')
    )
ak4TrackL2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4TrackL2Relative','ak4TrackL3Absolute')
    )

# L2L3Residual CORRECTORS
ak4CaloL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4CaloL2Relative','ak4CaloL3Absolute','ak4CaloResidual')
    )
ak4PFL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4PFL2Relative','ak4PFL3Absolute','ak4PFResidual')
    )
ak4PFCHSL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4PFCHSL2Relative','ak4PFCHSL3Absolute','ak4PFCHSResidual')
    )
#--- JPT needs the L1JPTOffset to account for the ZSP changes.
#--- L1JPTOffset is NOT the same as L1Offset !!!!!
ak4JPTL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4L1JPTOffset','ak4JPTL2Relative','ak4JPTL3Absolute','ak4JPTResidual')
    )

# L1L2L3 CORRECTORS
ak4CaloL1L2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4CaloL1Offset','ak4CaloL2Relative','ak4CaloL3Absolute')
    )
ak4PFL1L2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4PFL1Offset','ak4PFL2Relative','ak4PFL3Absolute')
    )
ak4PFCHSL1L2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4PFCHSL1Offset','ak4PFCHSL2Relative','ak4PFCHSL3Absolute')
    )
#--- JPT needs the L1JPTOffset to account for the ZSP changes.
#--- L1JPTOffset is NOT the same as L1Offset !!!!!
ak4JPTL1L2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4L1JPTOffset','ak4JPTL2Relative','ak4JPTL3Absolute')
    )

# L1L2L3Residual CORRECTORS
ak4CaloL1L2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4CaloL1Offset','ak4CaloL2Relative','ak4CaloL3Absolute','ak4CaloResidual')
    )
ak4PFL1L2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4PFL1Offset','ak4PFL2Relative','ak4PFL3Absolute','ak4PFResidual')
    )
ak4PFCHSL1L2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4PFCHSL1Offset','ak4PFCHSL2Relative','ak4PFCHSL3Absolute','ak4PFCHSResidual')
    )
#--- JPT needs the L1JPTOffset to account for the ZSP changes.
#--- L1JPTOffset is NOT the same as L1Offset !!!!!
ak4JPTL1L2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4L1JPTOffset','ak4JPTL2Relative','ak4JPTL3Absolute','ak4JPTResidual')
    )

# L1L2L3 CORRECTORS WITH FASTJET
ak4CaloL1FastL2L3Corrector = ak4CaloL2L3Corrector.clone()
ak4CaloL1FastL2L3Corrector.correctors.insert(0,'ak4CaloL1Fastjet')
ak4PFL1FastL2L3Corrector = ak4PFL2L3Corrector.clone()
ak4PFL1FastL2L3Corrector.correctors.insert(0,'ak4PFL1Fastjet')
ak4PFCHSL1FastL2L3Corrector = ak4PFCHSL2L3Corrector.clone()
ak4PFCHSL1FastL2L3Corrector.correctors.insert(0,'ak4PFCHSL1Fastjet')
#--- JPT needs the L1JPTOffset to account for the ZSP changes.
#--- L1JPTOffset is NOT the same as L1Offset !!!!!
ak4JPTL1FastL2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4JPTL1Fastjet','ak4JPTL2Relative','ak4JPTL3Absolute')
    )

# L1L2L3Residual CORRECTORS WITH FASTJET
ak4CaloL1FastL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4CaloL1Fastjet','ak4CaloL2Relative','ak4CaloL3Absolute','ak4CaloResidual')
    )
ak4PFL1FastL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4PFL1Fastjet','ak4PFL2Relative','ak4PFL3Absolute','ak4PFResidual')
    )
ak4PFCHSL1FastL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4PFCHSL1Fastjet','ak4PFCHSL2Relative','ak4PFCHSL3Absolute','ak4PFCHSResidual')
    )
#--- JPT needs the L1JPTOffset to account for the ZSP changes.
#--- L1JPTOffset is NOT the same as L1Offset !!!!!
ak4JPTL1FastL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4JPTL1Fastjet','ak4JPTL2Relative','ak4JPTL3Absolute','ak4JPTResidual')
    )

# L2L3L6 CORRECTORS
ak4CaloL2L3L6Corrector = ak4CaloL2L3Corrector.clone()
ak4CaloL2L3L6Corrector.correctors.append('ak4CaloL6SLB')
ak4PFL2L3L6Corrector = ak4PFL2L3Corrector.clone()
ak4PFL2L3L6Corrector.correctors.append('ak4PFL6SLB')

# L1L2L3L6 CORRECTORS
ak4CaloL1FastL2L3L6Corrector = ak4CaloL1FastL2L3Corrector.clone()
ak4CaloL1FastL2L3L6Corrector.correctors.append('ak4CaloL6SLB')
ak4PFL1FastL2L3L6Corrector = ak4PFL1FastL2L3Corrector.clone()
ak4PFL1FastL2L3L6Corrector.correctors.append('ak4PFL6SLB')
