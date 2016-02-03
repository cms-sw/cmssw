import FWCore.ParameterSet.Config as cms

#
# SINGLE LEVEL CORRECTORS
#

# L1 (Offset) CORRECTOR
ak4CaloL1OffsetCorrector = cms.EDProducer(
    'L1OffsetCorrectorProducer',
    level = cms.string('L1Offset'),
    algorithm = cms.string('AK5Calo'),
    vertexCollection = cms.InputTag('offlinePrimaryVertices'),
    minVtxNdof = cms.int32(4)
    )

ak4PFL1OffsetCorrector = ak4CaloL1OffsetCorrector.clone(algorithm = 'AK4PF')
ak4PFCHSL1OffsetCorrector = ak4CaloL1OffsetCorrector.clone(algorithm = 'AK4PFchs')
ak4JPTL1OffsetCorrector = ak4CaloL1OffsetCorrector.clone(algorithm = 'AK4JPT')
ak4PFPuppiL1OffsetCorrector = ak4CaloL1OffsetCorrector.clone(algorithm = 'AK4PFPuppi')

# L1 (JPT Offset) CORRECTOR
ak4L1JPTOffsetCorrector = cms.EDProducer(
    'L1JPTOffsetCorrectorProducer',
    level = cms.string('L1JPTOffset'),
    algorithm = cms.string('AK5JPT'),
    offsetService = cms.InputTag('ak4CaloL1OffsetCorrector')
    )
ak4L1JPTOffsetCorrectorChain = cms.Sequence(
    ak4CaloL1OffsetCorrector * ak4L1JPTOffsetCorrector
)

# L1 (Fastjet PU Subtraction) CORRECTOR
ak4CaloL1FastjetCorrector = cms.EDProducer(
    'L1FastjetCorrectorProducer',
    level       = cms.string('L1FastJet'),
    algorithm   = cms.string('AK5Calo'),
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
ak4PFPuppiL1FastjetCorrector = cms.EDProducer(
    'L1FastjetCorrectorProducer',
    level       = cms.string('L1FastJet'),
    algorithm   = cms.string('AK4PFPuppi'),
    srcRho      = cms.InputTag( 'fixedGridRhoFastjetAll' )
    )

# L2 (relative eta-conformity) CORRECTORS
ak4CaloL2RelativeCorrector = cms.EDProducer(
    'LXXXCorrectorProducer',
    level     = cms.string('L2Relative'),
    algorithm = cms.string('AK5Calo')
    )
ak4PFL2RelativeCorrector = ak4CaloL2RelativeCorrector.clone( algorithm = 'AK4PF' )
ak4PFCHSL2RelativeCorrector = ak4CaloL2RelativeCorrector.clone( algorithm = 'AK4PFchs' )
ak4JPTL2RelativeCorrector = ak4CaloL2RelativeCorrector.clone( algorithm = 'AK4JPT' )
ak4TrackL2RelativeCorrector = ak4CaloL2RelativeCorrector.clone( algorithm = 'AK4TRK' )
ak4PFPuppiL2RelativeCorrector = ak4CaloL2RelativeCorrector.clone( algorithm = 'AK4PFPuppi' )

# L3 (absolute) CORRECTORS
ak4CaloL3AbsoluteCorrector = cms.EDProducer(
    'LXXXCorrectorProducer',
    level     = cms.string('L3Absolute'),
    algorithm = cms.string('AK5Calo')
    )
ak4PFL3AbsoluteCorrector     = ak4CaloL3AbsoluteCorrector.clone( algorithm = 'AK4PF' )
ak4PFCHSL3AbsoluteCorrector     = ak4CaloL3AbsoluteCorrector.clone( algorithm = 'AK4PFchs' )
ak4JPTL3AbsoluteCorrector    = ak4CaloL3AbsoluteCorrector.clone( algorithm = 'AK4JPT' )
ak4TrackL3AbsoluteCorrector  = ak4CaloL3AbsoluteCorrector.clone( algorithm = 'AK4TRK' )
ak4PFPuppiL3AbsoluteCorrector     = ak4CaloL3AbsoluteCorrector.clone( algorithm = 'AK4PFPuppi' )

# Residual CORRECTORS
ak4CaloResidualCorrector = cms.EDProducer(
    'LXXXCorrectorProducer',
    level     = cms.string('L2L3Residual'),
    algorithm = cms.string('AK5Calo')
    )
ak4PFResidualCorrector  = ak4CaloResidualCorrector.clone( algorithm = 'AK4PF' )
ak4PFCHSResidualCorrector  = ak4CaloResidualCorrector.clone( algorithm = 'AK4PFchs' )
ak4JPTResidualCorrector = ak4CaloResidualCorrector.clone( algorithm = 'AK4JPT' )
ak4PFPuppiResidualCorrector  = ak4CaloResidualCorrector.clone( algorithm = 'AK4PFPuppi' )

# L6 (semileptonically decaying b-jet) Correction Services
ak4CaloL6SLBCorrector = cms.EDProducer(
    'L6SLBCorrectorProducer',
    level               = cms.string('L6SLB'),
    algorithm           = cms.string(''),
    addMuonToJet        = cms.bool(True),
    srcBTagInfoElectron = cms.InputTag('ak4CaloJetsSoftElectronTagInfos'),
    srcBTagInfoMuon     = cms.InputTag('ak4CaloJetsSoftMuonTagInfos')
    )
ak4PFL6SLBCorrector = cms.EDProducer(
    'L6SLBCorrectorProducer',
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
    correctors = cms.VInputTag('ak4CaloL2RelativeCorrector','ak4CaloL3AbsoluteCorrector')
    )
ak4CaloL2L3CorrectorChain = cms.Sequence(
    ak4CaloL2RelativeCorrector * ak4CaloL3AbsoluteCorrector * ak4CaloL2L3Corrector
)
ak4PFL2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4PFL2RelativeCorrector','ak4PFL3AbsoluteCorrector')
    )
ak4PFL2L3CorrectorChain = cms.Sequence(
    ak4PFL2RelativeCorrector * ak4PFL3AbsoluteCorrector * ak4PFL2L3Corrector
)
ak4PFCHSL2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4PFCHSL2RelativeCorrector','ak4PFCHSL3AbsoluteCorrector')
    )
ak4PFCHSL2L3CorrectorChain = cms.Sequence(
    ak4PFCHSL2RelativeCorrector * ak4PFCHSL3AbsoluteCorrector * ak4PFCHSL2L3Corrector
)
#--- JPT needs the L1JPTOffset to account for the ZSP changes.
#--- L1JPTOffset is NOT the same as L1Offset !!!!!
ak4JPTL2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4L1JPTOffsetCorrector','ak4JPTL2RelativeCorrector','ak4JPTL3AbsoluteCorrector')
    )
ak4JPTL2L3CorrectorChain = cms.Sequence(
    ak4L1JPTOffsetCorrectorChain * ak4JPTL2RelativeCorrector * ak4JPTL3AbsoluteCorrector * ak4JPTL2L3Corrector
)
ak4TrackL2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4TrackL2RelativeCorrector','ak4TrackL3AbsoluteCorrector')
    )
ak4TrackL2L3CorrectorChain = cms.Sequence(
    ak4TrackL2RelativeCorrector * ak4TrackL3AbsoluteCorrector * ak4TrackL2L3Corrector
)
ak4PFPuppiL2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4PFPuppiL2RelativeCorrector','ak4PFPuppiL3AbsoluteCorrector')
    )
ak4PFPuppiL2L3CorrectorChain = cms.Sequence(
    ak4PFPuppiL2RelativeCorrector * ak4PFPuppiL3AbsoluteCorrector * ak4PFPuppiL2L3Corrector
)

# L2L3Residual CORRECTORS
ak4CaloL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4CaloL2RelativeCorrector','ak4CaloL3AbsoluteCorrector','ak4CaloResidualCorrector')
    )
ak4CaloL2L3ResidualCorrectorChain = cms.Sequence(
    ak4CaloL2RelativeCorrector * ak4CaloL3AbsoluteCorrector * ak4CaloResidualCorrector * ak4CaloL2L3ResidualCorrector
)
ak4PFL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4PFL2RelativeCorrector','ak4PFL3AbsoluteCorrector','ak4PFResidualCorrector')
    )
ak4PFL2L3ResidualCorrectorChain = cms.Sequence(
    ak4PFL2RelativeCorrector * ak4PFL3AbsoluteCorrector * ak4PFResidualCorrector * ak4PFL2L3ResidualCorrector
)
ak4PFCHSL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4PFCHSL2RelativeCorrector','ak4PFCHSL3AbsoluteCorrector','ak4PFCHSResidualCorrector')
    )
ak4PFCHSL2L3ResidualCorrectorChain = cms.Sequence(
    ak4PFCHSL2RelativeCorrector * ak4PFCHSL3AbsoluteCorrector * ak4PFCHSResidualCorrector * ak4PFCHSL2L3ResidualCorrector
)
#--- JPT needs the L1JPTOffset to account for the ZSP changes.
#--- L1JPTOffset is NOT the same as L1Offset !!!!!
ak4JPTL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4L1JPTOffsetCorrector','ak4JPTL2RelativeCorrector','ak4JPTL3AbsoluteCorrector','ak4JPTResidualCorrector')
    )
ak4JPTL2L3ResidualCorrectorChain = cms.Sequence(
    ak4L1JPTOffsetCorrectorChain * ak4JPTL2RelativeCorrector * ak4JPTL3AbsoluteCorrector * ak4JPTResidualCorrector * ak4JPTL2L3ResidualCorrector
)
ak4PFPuppiL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4PFPuppiL2RelativeCorrector','ak4PFPuppiL3AbsoluteCorrector','ak4PFPuppiResidualCorrector')
    )
ak4PFPuppiL2L3ResidualCorrectorChain = cms.Sequence(
    ak4PFPuppiL2RelativeCorrector * ak4PFPuppiL3AbsoluteCorrector * ak4PFPuppiResidualCorrector * ak4PFPuppiL2L3ResidualCorrector
)

# L1L2L3 CORRECTORS
ak4CaloL1L2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4CaloL1OffsetCorrector','ak4CaloL2RelativeCorrector','ak4CaloL3AbsoluteCorrector')
    )
ak4CaloL1L2L3CorrectorChain = cms.Sequence(
    ak4CaloL1OffsetCorrector * ak4CaloL2RelativeCorrector * ak4CaloL3AbsoluteCorrector * ak4CaloL1L2L3Corrector
)
ak4PFL1L2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4PFL1OffsetCorrector','ak4PFL2RelativeCorrector','ak4PFL3AbsoluteCorrector')
    )
ak4PFL1L2L3CorrectorChain = cms.Sequence(
    ak4PFL1OffsetCorrector * ak4PFL2RelativeCorrector * ak4PFL3AbsoluteCorrector * ak4PFL1L2L3Corrector
)
ak4PFCHSL1L2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4PFCHSL1OffsetCorrector','ak4PFCHSL2RelativeCorrector','ak4PFCHSL3AbsoluteCorrector')
    )
ak4PFCHSL1L2L3CorrectorChain = cms.Sequence(
    ak4PFCHSL1OffsetCorrector * ak4PFCHSL2RelativeCorrector * ak4PFCHSL3AbsoluteCorrector * ak4PFCHSL1L2L3Corrector
)
#--- JPT needs the L1JPTOffset to account for the ZSP changes.
#--- L1JPTOffset is NOT the same as L1Offset !!!!!
ak4JPTL1L2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4L1JPTOffsetCorrector','ak4JPTL2RelativeCorrector','ak4JPTL3AbsoluteCorrector')
    )
ak4JPTL1L2L3CorrectorChain = cms.Sequence(
    ak4L1JPTOffsetCorrectorChain * ak4JPTL2RelativeCorrector * ak4JPTL3AbsoluteCorrector * ak4JPTL1L2L3Corrector
)
ak4PFPuppiL1L2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4PFPuppiL1OffsetCorrector','ak4PFPuppiL2RelativeCorrector','ak4PFPuppiL3AbsoluteCorrector')
    )
ak4PFPuppiL1L2L3CorrectorChain = cms.Sequence(
    ak4PFPuppiL1OffsetCorrector * ak4PFPuppiL2RelativeCorrector * ak4PFPuppiL3AbsoluteCorrector * ak4PFPuppiL1L2L3Corrector
)

# L1L2L3Residual CORRECTORS
ak4CaloL1L2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4CaloL1OffsetCorrector','ak4CaloL2RelativeCorrector','ak4CaloL3AbsoluteCorrector','ak4CaloResidualCorrector')
    )
ak4CaloL1L2L3ResidualCorrectorChain = cms.Sequence(
    ak4CaloL1OffsetCorrector * ak4CaloL2RelativeCorrector * ak4CaloL3AbsoluteCorrector * ak4CaloResidualCorrector * ak4CaloL1L2L3ResidualCorrector
)
ak4PFL1L2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4PFL1OffsetCorrector','ak4PFL2RelativeCorrector','ak4PFL3AbsoluteCorrector','ak4PFResidualCorrector')
    )
ak4PFL1L2L3ResidualCorrectorChain = cms.Sequence(
    ak4PFL1OffsetCorrector * ak4PFL2RelativeCorrector * ak4PFL3AbsoluteCorrector * ak4PFResidualCorrector * ak4PFL1L2L3ResidualCorrector
)
ak4PFCHSL1L2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4PFCHSL1OffsetCorrector','ak4PFCHSL2RelativeCorrector','ak4PFCHSL3AbsoluteCorrector','ak4PFCHSResidualCorrector')
    )
ak4PFCHSL1L2L3ResidualCorrectorChain = cms.Sequence(
    ak4PFCHSL1OffsetCorrector * ak4PFCHSL2RelativeCorrector * ak4PFCHSL3AbsoluteCorrector * ak4PFCHSResidualCorrector * ak4PFCHSL1L2L3ResidualCorrector
)
#--- JPT needs the L1JPTOffset to account for the ZSP changes.
#--- L1JPTOffset is NOT the same as L1Offset !!!!!
ak4JPTL1L2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4L1JPTOffsetCorrector','ak4JPTL2RelativeCorrector','ak4JPTL3AbsoluteCorrector','ak4JPTResidualCorrector')
    )
ak4JPTL1L2L3ResidualCorrectorChain = cms.Sequence(
    ak4L1JPTOffsetCorrectorChain * ak4JPTL2RelativeCorrector * ak4JPTL3AbsoluteCorrector * ak4JPTResidualCorrector * ak4JPTL1L2L3ResidualCorrector
)
ak4PFPuppiL1L2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4PFPuppiL1OffsetCorrector','ak4PFPuppiL2RelativeCorrector','ak4PFPuppiL3AbsoluteCorrector','ak4PFPuppiResidualCorrector')
    )
ak4PFPuppiL1L2L3ResidualCorrectorChain = cms.Sequence(
    ak4PFPuppiL1OffsetCorrector * ak4PFPuppiL2RelativeCorrector * ak4PFPuppiL3AbsoluteCorrector * ak4PFPuppiResidualCorrector * ak4PFPuppiL1L2L3ResidualCorrector
)

# L1L2L3 CORRECTORS WITH FASTJET
ak4CaloL1FastL2L3Corrector = ak4CaloL2L3Corrector.clone()
ak4CaloL1FastL2L3Corrector.correctors.insert(0,'ak4CaloL1FastjetCorrector')
ak4CaloL1FastL2L3CorrectorChain = cms.Sequence(
    ak4CaloL1FastjetCorrector * ak4CaloL2RelativeCorrector * ak4CaloL3AbsoluteCorrector * ak4CaloL1FastL2L3Corrector
)
ak4PFL1FastL2L3Corrector = ak4PFL2L3Corrector.clone()
ak4PFL1FastL2L3Corrector.correctors.insert(0,'ak4PFL1FastjetCorrector')
ak4PFL1FastL2L3CorrectorChain = cms.Sequence(
    ak4PFL1FastjetCorrector * ak4PFL2RelativeCorrector * ak4PFL3AbsoluteCorrector * ak4PFL1FastL2L3Corrector
)
ak4PFCHSL1FastL2L3Corrector = ak4PFCHSL2L3Corrector.clone()
ak4PFCHSL1FastL2L3Corrector.correctors.insert(0,'ak4PFCHSL1FastjetCorrector')
ak4PFCHSL1FastL2L3CorrectorChain = cms.Sequence(
    ak4PFCHSL1FastjetCorrector * ak4PFCHSL2RelativeCorrector * ak4PFCHSL3AbsoluteCorrector * ak4PFCHSL1FastL2L3Corrector
)
#--- JPT needs the L1JPTOffset to account for the ZSP changes.
#--- L1JPTOffset is NOT the same as L1Offset !!!!!
ak4JPTL1FastL2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4JPTL1FastjetCorrector','ak4JPTL2RelativeCorrector','ak4JPTL3AbsoluteCorrector')
    )
ak4JPTL1FastL2L3CorrectorChain = cms.Sequence(
    ak4JPTL1FastjetCorrector * ak4JPTL2RelativeCorrector * ak4JPTL3AbsoluteCorrector * ak4JPTL1FastL2L3Corrector
)
ak4PFPuppiL1FastL2L3Corrector = ak4PFPuppiL2L3Corrector.clone()
ak4PFPuppiL1FastL2L3Corrector.correctors.insert(0,'ak4PFPuppiL1FastjetCorrector')
ak4PFPuppiL1FastL2L3CorrectorChain = cms.Sequence(
    ak4PFPuppiL1FastjetCorrector * ak4PFPuppiL2RelativeCorrector * ak4PFPuppiL3AbsoluteCorrector * ak4PFPuppiL1FastL2L3Corrector
)

# L1L2L3Residual CORRECTORS WITH FASTJET
ak4CaloL1FastL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4CaloL1FastjetCorrector','ak4CaloL2RelativeCorrector','ak4CaloL3AbsoluteCorrector','ak4CaloResidualCorrector')
    )
ak4CaloL1FastL2L3ResidualCorrectorChain = cms.Sequence(
    ak4CaloL1FastjetCorrector * ak4CaloL2RelativeCorrector * ak4CaloL3AbsoluteCorrector * ak4CaloResidualCorrector * ak4CaloL1FastL2L3ResidualCorrector
)
ak4PFL1FastL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4PFL1FastjetCorrector','ak4PFL2RelativeCorrector','ak4PFL3AbsoluteCorrector','ak4PFResidualCorrector')
    )
ak4PFL1FastL2L3ResidualCorrectorChain = cms.Sequence(
    ak4PFL1FastjetCorrector * ak4PFL2RelativeCorrector * ak4PFL3AbsoluteCorrector * ak4PFResidualCorrector * ak4PFL1FastL2L3ResidualCorrector
)
ak4PFCHSL1FastL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4PFCHSL1FastjetCorrector','ak4PFCHSL2RelativeCorrector','ak4PFCHSL3AbsoluteCorrector','ak4PFCHSResidualCorrector')
    )
ak4PFCHSL1FastL2L3ResidualCorrectorChain = cms.Sequence(
    ak4PFCHSL1FastjetCorrector * ak4PFCHSL2RelativeCorrector * ak4PFCHSL3AbsoluteCorrector * ak4PFCHSResidualCorrector * ak4PFCHSL1FastL2L3ResidualCorrector
)
#--- JPT needs the L1JPTOffset to account for the ZSP changes.
#--- L1JPTOffset is NOT the same as L1Offset !!!!!
ak4JPTL1FastL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4JPTL1FastjetCorrector','ak4JPTL2RelativeCorrector','ak4JPTL3AbsoluteCorrector','ak4JPTResidualCorrector')
    )
ak4JPTL1FastL2L3ResidualCorrectorChain = cms.Sequence(
    ak4JPTL1FastjetCorrector * ak4JPTL2RelativeCorrector * ak4JPTL3AbsoluteCorrector * ak4JPTResidualCorrector * ak4JPTL1FastL2L3ResidualCorrector
)
ak4PFPuppiL1FastL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak4PFPuppiL1FastjetCorrector','ak4PFPuppiL2RelativeCorrector','ak4PFPuppiL3AbsoluteCorrector','ak4PFPuppiResidualCorrector')
    )
ak4PFPuppiL1FastL2L3ResidualCorrectorChain = cms.Sequence(
    ak4PFPuppiL1FastjetCorrector * ak4PFPuppiL2RelativeCorrector * ak4PFPuppiL3AbsoluteCorrector * ak4PFPuppiResidualCorrector * ak4PFPuppiL1FastL2L3ResidualCorrector
)

# L2L3L6 CORRECTORS
ak4CaloL2L3L6Corrector = ak4CaloL2L3Corrector.clone()
ak4CaloL2L3L6Corrector.correctors.append('ak4CaloL6SLBCorrector')
ak4CaloL2L3L6CorrectorChain = cms.Sequence(
    ak4CaloL2RelativeCorrector * ak4CaloL3AbsoluteCorrector * ak4CaloL6SLBCorrector * ak4CaloL2L3L6Corrector
)
ak4PFL2L3L6Corrector = ak4PFL2L3Corrector.clone()
ak4PFL2L3L6Corrector.correctors.append('ak4PFL6SLBCorrector')
ak4PFL2L3L6CorrectorChain = cms.Sequence(
    ak4PFL2RelativeCorrector * ak4PFL3AbsoluteCorrector * ak4PFL6SLBCorrector * ak4PFL2L3L6Corrector
)

# L1L2L3L6 CORRECTORS
ak4CaloL1FastL2L3L6Corrector = ak4CaloL1FastL2L3Corrector.clone()
ak4CaloL1FastL2L3L6Corrector.correctors.append('ak4CaloL6SLBCorrector')
ak4CaloL1FastL2L3L6CorrectorChain = cms.Sequence(
    ak4CaloL1FastjetCorrector * ak4CaloL2RelativeCorrector * ak4CaloL3AbsoluteCorrector * ak4CaloL6SLBCorrector * ak4CaloL1FastL2L3L6Corrector
)
ak4PFL1FastL2L3L6Corrector = ak4PFL1FastL2L3Corrector.clone()
ak4PFL1FastL2L3L6Corrector.correctors.append('ak4PFL6SLBCorrector')
ak4PFL1FastL2L3L6CorrectorChain = cms.Sequence(
    ak4PFL1FastjetCorrector * ak4PFL2RelativeCorrector * ak4PFL3AbsoluteCorrector * ak4PFL6SLBCorrector * ak4PFL1FastL2L3L6Corrector
)
