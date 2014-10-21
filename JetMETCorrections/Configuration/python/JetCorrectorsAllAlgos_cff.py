import FWCore.ParameterSet.Config as cms

from JetMETCorrections.Configuration.JetCorrectors_cff import *

#
# SINGLE LEVEL CORRECTORS
#

# L1 (offset) Correctors
ak7CaloL1OffsetCorrector = ak4CaloL1OffsetCorrector.clone()
kt4CaloL1OffsetCorrector = ak4CaloL1OffsetCorrector.clone()
kt6CaloL1OffsetCorrector = ak4CaloL1OffsetCorrector.clone()
ic5CaloL1OffsetCorrector = ak4CaloL1OffsetCorrector.clone()

ak1PFL1OffsetCorrector   = ak4PFL1OffsetCorrector.clone()
ak1PFCHSL1OffsetCorrector   = ak4PFCHSL1OffsetCorrector.clone()
ak2PFL1OffsetCorrector   = ak4PFL1OffsetCorrector.clone()
ak2PFCHSL1OffsetCorrector   = ak4PFCHSL1OffsetCorrector.clone()
ak3PFL1OffsetCorrector   = ak4PFL1OffsetCorrector.clone()
ak3PFCHSL1OffsetCorrector   = ak4PFCHSL1OffsetCorrector.clone()
ak5PFL1OffsetCorrector   = ak4PFL1OffsetCorrector.clone()
ak5PFCHSL1OffsetCorrector   = ak4PFCHSL1OffsetCorrector.clone()
ak6PFL1OffsetCorrector   = ak4PFL1OffsetCorrector.clone()
ak6PFCHSL1OffsetCorrector   = ak4PFCHSL1OffsetCorrector.clone()
ak7PFL1OffsetCorrector   = ak4PFL1OffsetCorrector.clone()
ak7PFCHSL1OffsetCorrector   = ak4PFCHSL1OffsetCorrector.clone()
ak8PFL1OffsetCorrector   = ak4PFL1OffsetCorrector.clone()
ak8PFCHSL1OffsetCorrector   = ak4PFCHSL1OffsetCorrector.clone()
ak9PFL1OffsetCorrector   = ak4PFL1OffsetCorrector.clone()
ak9PFCHSL1OffsetCorrector   = ak4PFCHSL1OffsetCorrector.clone()
ak10PFL1OffsetCorrector   = ak4PFL1OffsetCorrector.clone()
ak10PFCHSL1OffsetCorrector   = ak4PFCHSL1OffsetCorrector.clone()



kt4PFL1OffsetCorrector   = ak4PFL1OffsetCorrector.clone()
kt6PFL1OffsetCorrector   = ak4PFL1OffsetCorrector.clone()
ic5PFL1OffsetCorrector   = ak4PFL1OffsetCorrector.clone()

ak7JPTL1OffsetCorrector  = ak4CaloL1OffsetCorrector.clone()

# L1 (fastjet) Correctors
ak7CaloL1FastjetCorrector = ak4CaloL1FastjetCorrector.clone()
kt4CaloL1FastjetCorrector = ak4CaloL1FastjetCorrector.clone()
kt6CaloL1FastjetCorrector = ak4CaloL1FastjetCorrector.clone()
ic5CaloL1FastjetCorrector = ak4CaloL1FastjetCorrector.clone()

ak1PFL1FastjetCorrector   = ak4PFL1FastjetCorrector.clone()
ak1PFCHSL1FastjetCorrector   = ak4PFCHSL1FastjetCorrector.clone()
ak2PFL1FastjetCorrector   = ak4PFL1FastjetCorrector.clone()
ak2PFCHSL1FastjetCorrector   = ak4PFCHSL1FastjetCorrector.clone()
ak3PFL1FastjetCorrector   = ak4PFL1FastjetCorrector.clone()
ak3PFCHSL1FastjetCorrector   = ak4PFCHSL1FastjetCorrector.clone()
ak5PFL1FastjetCorrector   = ak4PFL1FastjetCorrector.clone()
ak5PFCHSL1FastjetCorrector   = ak4PFCHSL1FastjetCorrector.clone()
ak6PFL1FastjetCorrector   = ak4PFL1FastjetCorrector.clone()
ak6PFCHSL1FastjetCorrector   = ak4PFCHSL1FastjetCorrector.clone()
ak7PFL1FastjetCorrector   = ak4PFL1FastjetCorrector.clone()
ak7PFCHSL1FastjetCorrector   = ak4PFCHSL1FastjetCorrector.clone()
ak8PFL1FastjetCorrector   = ak4PFL1FastjetCorrector.clone()
ak8PFCHSL1FastjetCorrector   = ak4PFCHSL1FastjetCorrector.clone()
ak9PFL1FastjetCorrector   = ak4PFL1FastjetCorrector.clone()
ak9PFCHSL1FastjetCorrector   = ak4PFCHSL1FastjetCorrector.clone()
ak10PFL1FastjetCorrector   = ak4PFL1FastjetCorrector.clone()
ak10PFCHSL1FastjetCorrector   = ak4PFCHSL1FastjetCorrector.clone()
kt4PFL1FastjetCorrector   = ak4PFL1FastjetCorrector.clone()
kt6PFL1FastjetCorrector   = ak4PFL1FastjetCorrector.clone()
ic5PFL1FastjetCorrector   = ak4PFL1FastjetCorrector.clone()

ak7JPTL1FastjetCorrector  = ak4JPTL1FastjetCorrector.clone()

# SPECIAL L1JPTOffset
ak7L1JPTOffsetCorrector = ak4L1JPTOffsetCorrector.clone( offsetService = 'ak7CaloL1OffsetCorrector' )
ak7L1JPTOffsetCorrectorChain = cms.Sequence(
    ak7CaloL1OffsetCorrector * ak7L1JPTOffsetCorrector
)

# L2 (relative eta-conformity) Correctors
ak7CaloL2RelativeCorrector = ak4CaloL2RelativeCorrector.clone( algorithm = 'AK7Calo' )
ak7JPTL2RelativeCorrector = ak4CaloL2RelativeCorrector.clone( algorithm = 'AK7JPT' )
kt4CaloL2RelativeCorrector = ak4CaloL2RelativeCorrector.clone( algorithm = 'KT4Calo' )
kt6CaloL2RelativeCorrector = ak4CaloL2RelativeCorrector.clone( algorithm = 'KT6Calo' )
ic5CaloL2RelativeCorrector = ak4CaloL2RelativeCorrector.clone( algorithm = 'IC5Calo' )


ak1PFL2RelativeCorrector   = ak4PFL2RelativeCorrector.clone(algorithm='AK1PF')
ak1PFCHSL2RelativeCorrector   = ak4PFCHSL2RelativeCorrector.clone(algorithm='AK1PFchs')
ak2PFL2RelativeCorrector   = ak4PFL2RelativeCorrector.clone(algorithm='AK2PF')
ak2PFCHSL2RelativeCorrector   = ak4PFCHSL2RelativeCorrector.clone(algorithm='AK2PFchs')
ak3PFL2RelativeCorrector   = ak4PFL2RelativeCorrector.clone(algorithm='AK3PF')
ak3PFCHSL2RelativeCorrector   = ak4PFCHSL2RelativeCorrector.clone(algorithm='AK3PFchs')
ak5PFL2RelativeCorrector   = ak4PFL2RelativeCorrector.clone(algorithm='AK5PF')
ak5PFCHSL2RelativeCorrector   = ak4PFCHSL2RelativeCorrector.clone(algorithm='AK5PFchs')
ak6PFL2RelativeCorrector   = ak4PFL2RelativeCorrector.clone(algorithm='AK6PF')
ak6PFCHSL2RelativeCorrector   = ak4PFCHSL2RelativeCorrector.clone(algorithm='AK6PFchs')
ak7PFL2RelativeCorrector   = ak4PFL2RelativeCorrector.clone(algorithm='AK7PF')
ak7PFCHSL2RelativeCorrector   = ak4PFCHSL2RelativeCorrector.clone(algorithm='AK7PFchs')
ak8PFL2RelativeCorrector   = ak4PFL2RelativeCorrector.clone(algorithm='AK8PF')
ak8PFCHSL2RelativeCorrector   = ak4PFCHSL2RelativeCorrector.clone(algorithm='AK8PFchs')
ak9PFL2RelativeCorrector   = ak4PFL2RelativeCorrector.clone(algorithm='AK9PF')
ak9PFCHSL2RelativeCorrector   = ak4PFCHSL2RelativeCorrector.clone(algorithm='AK9PFchs')
ak10PFL2RelativeCorrector   = ak4PFL2RelativeCorrector.clone(algorithm='AK10PF')
ak10PFCHSL2RelativeCorrector   = ak4PFCHSL2RelativeCorrector.clone(algorithm='AK10PFchs')
kt4PFL2RelativeCorrector   = ak4PFL2RelativeCorrector.clone  ( algorithm = 'KT4PF' )
kt6PFL2RelativeCorrector   = ak4PFL2RelativeCorrector.clone  ( algorithm = 'KT6PF' )
ic5PFL2RelativeCorrector   = ak4PFL2RelativeCorrector.clone  ( algorithm = 'IC5PF' )

# L3 (absolute) Correctors
ak7CaloL3AbsoluteCorrector = ak4CaloL3AbsoluteCorrector.clone( algorithm = 'AK7Calo' )
ak7JPTL3AbsoluteCorrector = ak4CaloL3AbsoluteCorrector.clone( algorithm = 'AK7JPT' )
kt4CaloL3AbsoluteCorrector = ak4CaloL3AbsoluteCorrector.clone( algorithm = 'KT4Calo' )
kt6CaloL3AbsoluteCorrector = ak4CaloL3AbsoluteCorrector.clone( algorithm = 'KT6Calo' )
ic5CaloL3AbsoluteCorrector = ak4CaloL3AbsoluteCorrector.clone( algorithm = 'IC5Calo' )

ak1PFL3AbsoluteCorrector   = ak4PFL3AbsoluteCorrector.clone(algorithm='AK1PF')
ak1PFCHSL3AbsoluteCorrector   = ak4PFCHSL3AbsoluteCorrector.clone(algorithm='AK1PFchs')
ak2PFL3AbsoluteCorrector   = ak4PFL3AbsoluteCorrector.clone(algorithm='AK2PF')
ak2PFCHSL3AbsoluteCorrector   = ak4PFCHSL3AbsoluteCorrector.clone(algorithm='AK2PFchs')
ak3PFL3AbsoluteCorrector   = ak4PFL3AbsoluteCorrector.clone(algorithm='AK3PF')
ak3PFCHSL3AbsoluteCorrector   = ak4PFCHSL3AbsoluteCorrector.clone(algorithm='AK3PFchs')
ak5PFL3AbsoluteCorrector   = ak4PFL3AbsoluteCorrector.clone(algorithm='AK5PF')
ak5PFCHSL3AbsoluteCorrector   = ak4PFCHSL3AbsoluteCorrector.clone(algorithm='AK5PFchs')
ak6PFL3AbsoluteCorrector   = ak4PFL3AbsoluteCorrector.clone(algorithm='AK6PF')
ak6PFCHSL3AbsoluteCorrector   = ak4PFCHSL3AbsoluteCorrector.clone(algorithm='AK6PFchs')
ak7PFL3AbsoluteCorrector   = ak4PFL3AbsoluteCorrector.clone(algorithm='AK7PF')
ak7PFCHSL3AbsoluteCorrector   = ak4PFCHSL3AbsoluteCorrector.clone(algorithm='AK7PFchs')
ak8PFL3AbsoluteCorrector   = ak4PFL3AbsoluteCorrector.clone(algorithm='AK8PF')
ak8PFCHSL3AbsoluteCorrector   = ak4PFCHSL3AbsoluteCorrector.clone(algorithm='AK8PFchs')
ak9PFL3AbsoluteCorrector   = ak4PFL3AbsoluteCorrector.clone(algorithm='AK9PF')
ak9PFCHSL3AbsoluteCorrector   = ak4PFCHSL3AbsoluteCorrector.clone(algorithm='AK9PFchs')
ak10PFL3AbsoluteCorrector   = ak4PFL3AbsoluteCorrector.clone(algorithm='AK10PF')
ak10PFCHSL3AbsoluteCorrector   = ak4PFCHSL3AbsoluteCorrector.clone(algorithm='AK10PFchs')
kt4PFL3AbsoluteCorrector   = ak4PFL3AbsoluteCorrector.clone( algorithm = 'KT4PF' )
kt6PFL3AbsoluteCorrector   = ak4PFL3AbsoluteCorrector.clone( algorithm = 'KT6PF' )
ic5PFL3AbsoluteCorrector   = ak4PFL3AbsoluteCorrector.clone( algorithm = 'IC5PF' )

# Residual Correctors
ak7CaloResidualCorrector   = ak4CaloResidualCorrector.clone()
ak7JPTResidualCorrector   = ak4CaloResidualCorrector.clone()
kt4CaloResidualCorrector   = ak4CaloResidualCorrector.clone()
kt6CaloResidualCorrector   = ak4CaloResidualCorrector.clone()
ic5CaloResidualCorrector   = ak4CaloResidualCorrector.clone()

ak1PFResidualCorrector   = ak4PFResidualCorrector.clone()
ak1PFCHSResidualCorrector   = ak4PFCHSResidualCorrector.clone()
ak2PFResidualCorrector   = ak4PFResidualCorrector.clone()
ak2PFCHSResidualCorrector   = ak4PFCHSResidualCorrector.clone()
ak3PFResidualCorrector   = ak4PFResidualCorrector.clone()
ak3PFCHSResidualCorrector   = ak4PFCHSResidualCorrector.clone()
ak5PFResidualCorrector   = ak4PFResidualCorrector.clone()
ak5PFCHSResidualCorrector   = ak4PFCHSResidualCorrector.clone()
ak6PFResidualCorrector   = ak4PFResidualCorrector.clone()
ak6PFCHSResidualCorrector   = ak4PFCHSResidualCorrector.clone()
ak7PFResidualCorrector   = ak4PFResidualCorrector.clone()
ak7PFCHSResidualCorrector   = ak4PFCHSResidualCorrector.clone()
ak8PFResidualCorrector   = ak4PFResidualCorrector.clone()
ak8PFCHSResidualCorrector   = ak4PFCHSResidualCorrector.clone()
ak9PFResidualCorrector   = ak4PFResidualCorrector.clone()
ak9PFCHSResidualCorrector   = ak4PFCHSResidualCorrector.clone()
ak10PFResidualCorrector   = ak4PFResidualCorrector.clone()
ak10PFCHSResidualCorrector   = ak4PFCHSResidualCorrector.clone()
kt4PFResidualCorrector   = ak4PFResidualCorrector.clone()
kt6PFResidualCorrector   = ak4PFResidualCorrector.clone()
ic5PFResidualCorrector   = ak4PFResidualCorrector.clone()

# L6 (semileptonically decaying b-jet) Correctors
ak7CaloL6SLBCorrector = ak4CaloL6SLBCorrector.clone(
    srcBTagInfoElectron = cms.InputTag('ak7CaloJetsSoftElectronTagInfos'),
    srcBTagInfoMuon     = cms.InputTag('ak7CaloJetsSoftMuonTagInfos')
    )
ak7JPTL6SLBCorrector = ak4CaloL6SLBCorrector.clone(
    srcBTagInfoElectron = cms.InputTag('ak7JPTJetsSoftElectronTagInfos'),
    srcBTagInfoMuon     = cms.InputTag('ak7JPTJetsSoftMuonTagInfos')
    )
kt4CaloL6SLBCorrector = ak4CaloL6SLBCorrector.clone(
    srcBTagInfoElectron = cms.InputTag('kt4CaloJetsSoftElectronTagInfos'),
    srcBTagInfoMuon     = cms.InputTag('kt4CaloJetsSoftMuonTagInfos')
    )
kt6CaloL6SLBCorrector = ak4CaloL6SLBCorrector.clone(
    srcBTagInfoElectron = cms.InputTag('kt6CaloJetsSoftElectronTagInfos'),
    srcBTagInfoMuon     = cms.InputTag('kt6CaloJetsSoftMuonTagInfos')
    )
ic5CaloL6SLBCorrector = ak4CaloL6SLBCorrector.clone(
    srcBTagInfoElectron = cms.InputTag('ic5CaloJetsSoftElectronTagInfos'),
    srcBTagInfoMuon     = cms.InputTag('ic5CaloJetsSoftMuonTagInfos')
    )

ak7PFL6SLBCorrector = ak4PFL6SLBCorrector.clone(
    srcBTagInfoElectron = cms.InputTag('ak7PFJetsSoftElectronTagInfos'),
    srcBTagInfoMuon     = cms.InputTag('ak7PFJetsSoftMuonTagInfos')
    )
kt4PFL6SLBCorrector = ak4PFL6SLBCorrector.clone(
    srcBTagInfoElectron = cms.InputTag('kt4PFJetsSoftElectronTagInfos'),
    srcBTagInfoMuon     = cms.InputTag('kt4PFJetsSoftMuonTagInfos')
    )
kt6PFL6SLBCorrector = ak4PFL6SLBCorrector.clone(
    srcBTagInfoElectron = cms.InputTag('kt6PFJetsSoftElectronTagInfos'),
    srcBTagInfoMuon     = cms.InputTag('kt6PFJetsSoftMuonTagInfos')
    )
ic5PFL6SLBCorrector = ak4PFL6SLBCorrector.clone(
    srcBTagInfoElectron = cms.InputTag('ic5PFJetsSoftElectronTagInfos'),
    srcBTagInfoMuon     = cms.InputTag('ic5PFJetsSoftMuonTagInfos')
    )


#
# MULTIPLE LEVEL CORRECTORS
#

# L2L3 CORRECTORS
ak7CaloL2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak7CaloL2RelativeCorrector','ak7CaloL3AbsoluteCorrector')
    )
ak7CaloL2L3CorrectorChain = cms.Sequence(
    ak7CaloL2RelativeCorrector * ak7CaloL3AbsoluteCorrector * ak7CaloL2L3Corrector
)
kt4CaloL2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('kt4CaloL2RelativeCorrector','kt4CaloL3AbsoluteCorrector')
    )
kt4CaloL2L3CorrectorChain = cms.Sequence(
    kt4CaloL2RelativeCorrector * kt4CaloL3AbsoluteCorrector * kt4CaloL2L3Corrector
)
kt6CaloL2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('kt6CaloL2RelativeCorrector','kt6CaloL3AbsoluteCorrector')
    )
kt6CaloL2L3CorrectorChain = cms.Sequence(
    kt6CaloL2RelativeCorrector * kt6CaloL3AbsoluteCorrector * kt6CaloL2L3Corrector
)
ic5CaloL2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ic5CaloL2RelativeCorrector','ic5CaloL3AbsoluteCorrector')
    )
ic5CaloL2L3CorrectorChain = cms.Sequence(
    ic5CaloL2RelativeCorrector * ic5CaloL3AbsoluteCorrector * ic5CaloL2L3Corrector
)


ak1PFL2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak1PFL2RelativeCorrector','ak1PFL3AbsoluteCorrector')
    )
ak1PFL2L3CorrectorChain = cms.Sequence(
    ak1PFL2RelativeCorrector * ak1PFL3AbsoluteCorrector * ak1PFL2L3Corrector
)

ak1PFCHSL2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak1PFCHSL2RelativeCorrector','ak1PFCHSL3AbsoluteCorrector')
    )
ak1PFCHSL2L3CorrectorChain = cms.Sequence(
    ak1PFCHSL2RelativeCorrector * ak1PFCHSL3AbsoluteCorrector * ak1PFCHSL2L3Corrector
)

ak2PFL2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak2PFL2RelativeCorrector','ak2PFL3AbsoluteCorrector')
    )
ak2PFL2L3CorrectorChain = cms.Sequence(
    ak2PFL2RelativeCorrector * ak2PFL3AbsoluteCorrector * ak2PFL2L3Corrector
)

ak2PFCHSL2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak2PFCHSL2RelativeCorrector','ak2PFCHSL3AbsoluteCorrector')
    )
ak2PFCHSL2L3CorrectorChain = cms.Sequence(
    ak2PFCHSL2RelativeCorrector * ak2PFCHSL3AbsoluteCorrector * ak2PFCHSL2L3Corrector
)

ak3PFL2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak3PFL2RelativeCorrector','ak3PFL3AbsoluteCorrector')
    )
ak3PFL2L3CorrectorChain = cms.Sequence(
    ak3PFL2RelativeCorrector * ak3PFL3AbsoluteCorrector * ak3PFL2L3Corrector
)

ak3PFCHSL2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak3PFCHSL2RelativeCorrector','ak3PFCHSL3AbsoluteCorrector')
    )
ak3PFCHSL2L3CorrectorChain = cms.Sequence(
    ak3PFCHSL2RelativeCorrector * ak3PFCHSL3AbsoluteCorrector * ak3PFCHSL2L3Corrector
)

ak5PFL2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak5PFL2RelativeCorrector','ak5PFL3AbsoluteCorrector')
    )
ak5PFL2L3CorrectorChain = cms.Sequence(
    ak5PFL2RelativeCorrector * ak5PFL3AbsoluteCorrector * ak5PFL2L3Corrector
)

ak5PFCHSL2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak5PFCHSL2RelativeCorrector','ak5PFCHSL3AbsoluteCorrector')
    )
ak5PFCHSL2L3CorrectorChain = cms.Sequence(
    ak5PFCHSL2RelativeCorrector * ak5PFCHSL3AbsoluteCorrector * ak5PFCHSL2L3Corrector
)

ak6PFL2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak6PFL2RelativeCorrector','ak6PFL3AbsoluteCorrector')
    )
ak6PFL2L3CorrectorChain = cms.Sequence(
    ak6PFL2RelativeCorrector * ak6PFL3AbsoluteCorrector * ak6PFL2L3Corrector
)

ak6PFCHSL2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak6PFCHSL2RelativeCorrector','ak6PFCHSL3AbsoluteCorrector')
    )
ak6PFCHSL2L3CorrectorChain = cms.Sequence(
    ak6PFCHSL2RelativeCorrector * ak6PFCHSL3AbsoluteCorrector * ak6PFCHSL2L3Corrector
)

ak7PFL2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak7PFL2RelativeCorrector','ak7PFL3AbsoluteCorrector')
    )
ak7PFL2L3CorrectorChain = cms.Sequence(
    ak7PFL2RelativeCorrector * ak7PFL3AbsoluteCorrector * ak7PFL2L3Corrector
)

ak7PFCHSL2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak7PFCHSL2RelativeCorrector','ak7PFCHSL3AbsoluteCorrector')
    )
ak7PFCHSL2L3CorrectorChain = cms.Sequence(
    ak7PFCHSL2RelativeCorrector * ak7PFCHSL3AbsoluteCorrector * ak7PFCHSL2L3Corrector
)

ak8PFL2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak8PFL2RelativeCorrector','ak8PFL3AbsoluteCorrector')
    )
ak8PFL2L3CorrectorChain = cms.Sequence(
    ak8PFL2RelativeCorrector * ak8PFL3AbsoluteCorrector * ak8PFL2L3Corrector
)

ak8PFCHSL2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak8PFCHSL2RelativeCorrector','ak8PFCHSL3AbsoluteCorrector')
    )
ak8PFCHSL2L3CorrectorChain = cms.Sequence(
    ak8PFCHSL2RelativeCorrector * ak8PFCHSL3AbsoluteCorrector * ak8PFCHSL2L3Corrector
)

ak9PFL2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak9PFL2RelativeCorrector','ak9PFL3AbsoluteCorrector')
    )
ak9PFL2L3CorrectorChain = cms.Sequence(
     ak9PFL2RelativeCorrector * ak9PFL3AbsoluteCorrector * ak9PFL2L3Corrector
)

ak9PFCHSL2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak9PFCHSL2RelativeCorrector','ak9PFCHSL3AbsoluteCorrector')
    )
ak9PFCHSL2L3CorrectorChain = cms.Sequence(
    ak9PFCHSL2RelativeCorrector * ak9PFCHSL3AbsoluteCorrector * ak9PFCHSL2L3Corrector
)

ak10PFL2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak10PFL2RelativeCorrector','ak10PFL3AbsoluteCorrector')
    )
ak10PFL2L3CorrectorChain = cms.Sequence(
    ak10PFL2RelativeCorrector * ak10PFL3AbsoluteCorrector * ak10PFL2L3Corrector
)

ak10PFCHSL2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak10PFCHSL2RelativeCorrector','ak10PFCHSL3AbsoluteCorrector')
    )
ak10PFCHSL2L3CorrectorChain = cms.Sequence(
    ak10PFCHSL2RelativeCorrector * ak10PFCHSL3AbsoluteCorrector * ak10PFCHSL2L3Corrector
)

kt4PFL2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('kt4PFL2RelativeCorrector','kt4PFL3AbsoluteCorrector')
    )
kt4PFL2L3CorrectorChain = cms.Sequence(
    kt4PFL2RelativeCorrector * kt4PFL3AbsoluteCorrector * kt4PFL2L3Corrector
)
kt6PFL2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('kt6PFL2RelativeCorrector','kt6PFL3AbsoluteCorrector')
    )
kt6PFL2L3CorrectorChain = cms.Sequence(
    kt6PFL2RelativeCorrector * kt6PFL3AbsoluteCorrector * kt6PFL2L3Corrector
)
ic5PFL2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ic5PFL2RelativeCorrector','ic5PFL3AbsoluteCorrector')
    )
ic5PFL2L3CorrectorChain = cms.Sequence(
    ic5PFL2RelativeCorrector * ic5PFL3AbsoluteCorrector * ic5PFL2L3Corrector
)

#--- JPT needs the L1JPTOffset to account for the ZSP changes.
#--- L1JPTOffset is NOT the same as L1Offset !!!!!
ak7JPTL2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak7L1JPTOffsetCorrector','ak7JPTL2RelativeCorrector','ak7JPTL3AbsoluteCorrector')
    )
ak7JPTL2L3CorrectorChain = cms.Sequence(
    ak7L1JPTOffsetCorrectorChain * ak7JPTL2RelativeCorrector * ak7JPTL3AbsoluteCorrector * ak7JPTL2L3Corrector
)

# L1L2L3 CORRECTORS
ak7CaloL1L2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak7CaloL1OffsetCorrector','ak7CaloL2RelativeCorrector','ak7CaloL3AbsoluteCorrector')
    )
ak7CaloL1L2L3CorrectorChain = cms.Sequence(
    ak7CaloL1OffsetCorrector * ak7CaloL2RelativeCorrector * ak7CaloL3AbsoluteCorrector * ak7CaloL1L2L3Corrector
)
kt4CaloL1L2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('kt4CaloL1OffsetCorrector','kt4CaloL2RelativeCorrector','kt4CaloL3AbsoluteCorrector')
    )
kt4CaloL1L2L3CorrectorChain = cms.Sequence(
    kt4CaloL1OffsetCorrector * kt4CaloL2RelativeCorrector * kt4CaloL3AbsoluteCorrector * kt4CaloL1L2L3Corrector
)
kt6CaloL1L2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('kt6CaloL1OffsetCorrector','kt6CaloL2RelativeCorrector','kt6CaloL3AbsoluteCorrector')
    )
kt6CaloL1L2L3CorrectorChain = cms.Sequence(
    kt6CaloL1OffsetCorrector * kt6CaloL2RelativeCorrector * kt6CaloL3AbsoluteCorrector * kt6CaloL1L2L3Corrector
)
ic5CaloL1L2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ic5CaloL1OffsetCorrector','ic5CaloL2RelativeCorrector','ic5CaloL3AbsoluteCorrector')
    )
ic5CaloL1L2L3CorrectorChain = cms.Sequence(
    ic5CaloL1OffsetCorrector * ic5CaloL2RelativeCorrector * ic5CaloL3AbsoluteCorrector * ic5CaloL1L2L3Corrector
)

ak7PFL1L2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak7PFL1OffsetCorrector','ak7PFL2RelativeCorrector','ak7PFL3AbsoluteCorrector')
    )
ak7PFL1L2L3CorrectorChain = cms.Sequence(
    ak7PFL1OffsetCorrector * ak7PFL2RelativeCorrector * ak7PFL3AbsoluteCorrector * ak7PFL1L2L3Corrector
)
kt4PFL1L2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('kt4PFL1OffsetCorrector','kt4PFL2RelativeCorrector','kt4PFL3AbsoluteCorrector')
    )
kt4PFL1L2L3CorrectorChain = cms.Sequence(
    kt4PFL1OffsetCorrector * kt4PFL2RelativeCorrector * kt4PFL3AbsoluteCorrector * kt4PFL1L2L3Corrector
)
kt6PFL1L2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('kt6PFL1OffsetCorrector','kt6PFL2RelativeCorrector','kt6PFL3AbsoluteCorrector')
    )
kt6PFL1L2L3CorrectorChain = cms.Sequence(
    kt6PFL1OffsetCorrector * kt6PFL2RelativeCorrector * kt6PFL3AbsoluteCorrector * kt6PFL1L2L3Corrector
)
ic5PFL1L2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ic5PFL1OffsetCorrector','ic5PFL2RelativeCorrector','ic5PFL3AbsoluteCorrector')
    )
ic5PFL1L2L3CorrectorChain = cms.Sequence(
    ic5PFL1OffsetCorrector * ic5PFL2RelativeCorrector * ic5PFL3AbsoluteCorrector * ic5PFL1L2L3Corrector
)
#--- JPT needs the L1JPTOffset to account for the ZSP changes.
#--- L1JPTOffset is NOT the same as L1Offset !!!!!
ak7JPTL1L2L3Corrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak7JPTL1OffsetCorrector','ak7L1JPTOffsetCorrector','ak7JPTL2RelativeCorrector','ak7JPTL3AbsoluteCorrector')
    )
ak7JPTL1L2L3CorrectorChain = cms.Sequence(
    ak7JPTL1OffsetCorrector * ak7L1JPTOffsetCorrectorChain * ak7JPTL2RelativeCorrector * ak7JPTL3AbsoluteCorrector * ak7JPTL1L2L3Corrector
)

# L2L3Residual CORRECTORS
ak7CaloL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak7CaloL2RelativeCorrector','ak7CaloL3AbsoluteCorrector','ak7CaloResidualCorrector')
    )
ak7CaloL2L3ResidualCorrectorChain = cms.Sequence(
    ak7CaloL2RelativeCorrector * ak7CaloL3AbsoluteCorrector * ak7CaloResidualCorrector * ak7CaloL2L3ResidualCorrector
)
kt4CaloL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('kt4CaloL2RelativeCorrector','kt4CaloL3AbsoluteCorrector','kt4CaloResidualCorrector')
    )
kt4CaloL2L3ResidualCorrectorChain = cms.Sequence(
    kt4CaloL2RelativeCorrector * kt4CaloL3AbsoluteCorrector * kt4CaloResidualCorrector * kt4CaloL2L3ResidualCorrector
)
kt6CaloL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('kt6CaloL2RelativeCorrector','kt6CaloL3AbsoluteCorrector','kt6CaloResidualCorrector')
    )
kt6CaloL2L3ResidualCorrectorChain = cms.Sequence(
    kt6CaloL2RelativeCorrector * kt6CaloL3AbsoluteCorrector * kt6CaloResidualCorrector * kt6CaloL2L3ResidualCorrector
)
ic5CaloL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ic5CaloL2RelativeCorrector','ic5CaloL3AbsoluteCorrector','ic5CaloResidualCorrector')
    )
ic5CaloL2L3ResidualCorrectorChain = cms.Sequence(
    ic5CaloL2RelativeCorrector * ic5CaloL3AbsoluteCorrector * ic5CaloResidualCorrector * ic5CaloL2L3ResidualCorrector
)





ak1PFL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak1PFL2RelativeCorrector','ak1PFL3AbsoluteCorrector','ak1PFResidualCorrector')
    )
ak1PFL2L3ResidualCorrectorChain = cms.Sequence(
    ak1PFL2RelativeCorrector * ak1PFL3AbsoluteCorrector * ak1PFResidualCorrector * ak1PFL2L3ResidualCorrector
)
ak1PFCHSL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak1PFCHSL2RelativeCorrector','ak1PFCHSL3AbsoluteCorrector','ak1PFCHSResidualCorrector')
    )
ak1PFCHSL2L3ResidualCorrectorChain = cms.Sequence(
    ak1PFCHSL2RelativeCorrector * ak1PFCHSL3AbsoluteCorrector * ak1PFCHSResidualCorrector * ak1PFCHSL2L3ResidualCorrector
)
ak2PFL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak2PFL2RelativeCorrector','ak2PFL3AbsoluteCorrector','ak2PFResidualCorrector')
    )
Chain = cms.Sequence(
    ak2PFL2RelativeCorrector * ak2PFL3AbsoluteCorrector * ak2PFResidualCorrector * ak2PFL2L3ResidualCorrector
)
ak2PFCHSL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak2PFCHSL2RelativeCorrector','ak2PFCHSL3AbsoluteCorrector','ak2PFCHSResidualCorrector')
    )
ak2PFCHSL2L3ResidualCorrectorChain = cms.Sequence(
    ak2PFCHSL2RelativeCorrector * ak2PFCHSL3AbsoluteCorrector * ak2PFCHSResidualCorrector * ak2PFCHSL2L3ResidualCorrector
)
ak3PFL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak3PFL2RelativeCorrector','ak3PFL3AbsoluteCorrector','ak3PFResidualCorrector')
    )
ak3PFL2L3ResidualCorrectorChain = cms.Sequence(
    ak3PFL2RelativeCorrector * ak3PFL3AbsoluteCorrector * ak3PFResidualCorrector * ak3PFL2L3ResidualCorrector
)
ak3PFCHSL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak3PFCHSL2RelativeCorrector','ak3PFCHSL3AbsoluteCorrector','ak3PFCHSResidualCorrector')
    )
ak3PFCHSL2L3ResidualCorrectorChain = cms.Sequence(
    ak3PFCHSL2RelativeCorrector * ak3PFCHSL3AbsoluteCorrector * ak3PFCHSResidualCorrector * ak3PFCHSL2L3ResidualCorrector
)
ak5PFL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak5PFL2RelativeCorrector','ak5PFL3AbsoluteCorrector','ak5PFResidualCorrector')
    )
ak5PFL2L3ResidualCorrectorChain = cms.Sequence(
    ak5PFL2RelativeCorrector * ak5PFL3AbsoluteCorrector * ak5PFResidualCorrector * ak5PFL2L3ResidualCorrector
)
ak5PFCHSL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak5PFCHSL2RelativeCorrector','ak5PFCHSL3AbsoluteCorrector','ak5PFCHSResidualCorrector')
    )
ak5PFCHSL2L3ResidualCorrectorChain = cms.Sequence(
    ak5PFCHSL2RelativeCorrector * ak5PFCHSL3AbsoluteCorrector * ak5PFCHSResidualCorrector * ak5PFCHSL2L3ResidualCorrector
)
ak6PFL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak6PFL2RelativeCorrector','ak6PFL3AbsoluteCorrector','ak6PFResidualCorrector')
    )
ak6PFL2L3ResidualCorrectorChain = cms.Sequence(
    ak6PFL2RelativeCorrector * ak6PFL3AbsoluteCorrector * ak6PFResidualCorrector * ak6PFL2L3ResidualCorrector
)
ak6PFCHSL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak6PFCHSL2RelativeCorrector','ak6PFCHSL3AbsoluteCorrector','ak6PFCHSResidualCorrector')
    )
ak6PFCHSL2L3ResidualCorrectorChain = cms.Sequence(
    ak6PFCHSL2RelativeCorrector * ak6PFCHSL3AbsoluteCorrector * ak6PFCHSResidualCorrector * ak6PFCHSL2L3ResidualCorrector
)
ak7PFL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak7PFL2RelativeCorrector','ak7PFL3AbsoluteCorrector','ak7PFResidualCorrector')
    )
ak7PFL2L3ResidualCorrectorChain = cms.Sequence(
    ak7PFL2RelativeCorrector * ak7PFL3AbsoluteCorrector * ak7PFResidualCorrector * ak7PFL2L3ResidualCorrector
)
ak7PFCHSL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak7PFCHSL2RelativeCorrector','ak7PFCHSL3AbsoluteCorrector','ak7PFCHSResidualCorrector')
    )
ak7PFCHSL2L3ResidualCorrectorChain = cms.Sequence(
    ak7PFCHSL2RelativeCorrector * ak7PFCHSL3AbsoluteCorrector * ak7PFCHSResidualCorrector * ak7PFCHSL2L3ResidualCorrector
)
ak8PFL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak8PFL2RelativeCorrector','ak8PFL3AbsoluteCorrector','ak8PFResidualCorrector')
    )
ak8PFL2L3ResidualCorrectorChain = cms.Sequence(
    ak8PFL2RelativeCorrector * ak8PFL3AbsoluteCorrector * ak8PFResidualCorrector * ak8PFL2L3ResidualCorrector
)
ak8PFCHSL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak8PFCHSL2RelativeCorrector','ak8PFCHSL3AbsoluteCorrector','ak8PFCHSResidualCorrector')
    )
ak8PFCHSL2L3ResidualCorrectorChain = cms.Sequence(
    ak8PFCHSL2RelativeCorrector * ak8PFCHSL3AbsoluteCorrector * ak8PFCHSResidualCorrector * ak8PFCHSL2L3ResidualCorrector
)
ak9PFL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak9PFL2RelativeCorrector','ak9PFL3AbsoluteCorrector','ak9PFResidualCorrector')
    )
ak9PFL2L3ResidualCorrectorChain = cms.Sequence(
    ak9PFL2RelativeCorrector * ak9PFL3AbsoluteCorrector * ak9PFResidualCorrector * ak9PFL2L3ResidualCorrector
)
ak9PFCHSL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak9PFCHSL2RelativeCorrector','ak9PFCHSL3AbsoluteCorrector','ak9PFCHSResidualCorrector')
    )
ak9PFCHSL2L3ResidualCorrectorChain = cms.Sequence(
    ak9PFCHSL2RelativeCorrector * ak9PFCHSL3AbsoluteCorrector * ak9PFCHSResidualCorrector * ak9PFCHSL2L3ResidualCorrector
)
ak10PFL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak10PFL2RelativeCorrector','ak10PFL3AbsoluteCorrector','ak10PFResidualCorrector')
    )
ak10PFL2L3ResidualCorrectorChain = cms.Sequence(
    ak10PFL2RelativeCorrector * ak10PFL3AbsoluteCorrector * ak10PFResidualCorrector * ak10PFL2L3ResidualCorrector
)
ak10PFCHSL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak10PFCHSL2RelativeCorrector','ak10PFCHSL3AbsoluteCorrector','ak10PFCHSResidualCorrector')
    )
ak10PFCHSL2L3ResidualCorrectorChain = cms.Sequence(
    ak10PFCHSL2RelativeCorrector * ak10PFCHSL3AbsoluteCorrector * ak10PFCHSResidualCorrector * ak10PFCHSL2L3ResidualCorrector
)

kt4PFL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('kt4PFL2RelativeCorrector','kt4PFL3AbsoluteCorrector','kt4PFResidualCorrector')
    )
kt4PFL2L3ResidualCorrectorChain = cms.Sequence(
    kt4PFL2RelativeCorrector * kt4PFL3AbsoluteCorrector * kt4PFResidualCorrector * kt4PFL2L3ResidualCorrector
)
kt6PFL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('kt6PFL2RelativeCorrector','kt6PFL3AbsoluteCorrector','kt6PFResidualCorrector')
    )
kt6PFL2L3ResidualCorrectorChain = cms.Sequence(
    kt6PFL2RelativeCorrector * kt6PFL3AbsoluteCorrector * kt6PFResidualCorrector * kt6PFL2L3ResidualCorrector
)
ic5PFL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ic5PFL2RelativeCorrector','ic5PFL3AbsoluteCorrector','ic5PFResidualCorrector')
    )
ic5PFL2L3ResidualCorrectorChain = cms.Sequence(
    ic5PFL2RelativeCorrector * ic5PFL3AbsoluteCorrector * ic5PFResidualCorrector * ic5PFL2L3ResidualCorrector
)

# L1L2L3Residual CORRECTORS
ak7CaloL1L2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak7CaloL1OffsetCorrector','ak7CaloL2RelativeCorrector','ak7CaloL3AbsoluteCorrector','ak7CaloResidualCorrector')
    )
ak7CaloL1L2L3ResidualCorrectorChain = cms.Sequence(
    ak7CaloL1OffsetCorrector * ak7CaloL2RelativeCorrector * ak7CaloL3AbsoluteCorrector * ak7CaloResidualCorrector * ak7CaloL1L2L3ResidualCorrector
)
kt4CaloL1L2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('kt4CaloL1OffsetCorrector','kt4CaloL2RelativeCorrector','kt4CaloL3AbsoluteCorrector','kt4CaloResidualCorrector')
    )
kt4CaloL1L2L3ResidualCorrectorChain = cms.Sequence(
    kt4CaloL1OffsetCorrector * kt4CaloL2RelativeCorrector * kt4CaloL3AbsoluteCorrector * kt4CaloResidualCorrector * kt4CaloL1L2L3ResidualCorrector
)
kt6CaloL1L2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('kt6CaloL1OffsetCorrector','kt6CaloL2RelativeCorrector','kt6CaloL3AbsoluteCorrector','kt6CaloResidualCorrector')
    )
kt6CaloL1L2L3ResidualCorrectorChain = cms.Sequence(
    kt6CaloL1OffsetCorrector * kt6CaloL2RelativeCorrector * kt6CaloL3AbsoluteCorrector * kt6CaloResidualCorrector * kt6CaloL1L2L3ResidualCorrector
)
ic5CaloL1L2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ic5CaloL1OffsetCorrector','ic5CaloL2RelativeCorrector','ic5CaloL3AbsoluteCorrector','ic5CaloResidualCorrector')
    )
ic5CaloL1L2L3ResidualCorrectorChain = cms.Sequence(
    ic5CaloL1OffsetCorrector * ic5CaloL2RelativeCorrector * ic5CaloL3AbsoluteCorrector * ic5CaloResidualCorrector * ic5CaloL1L2L3ResidualCorrector
)

ak1PFL1L2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak1PFL1OffsetCorrector','ak1PFL2RelativeCorrector','ak1PFL3AbsoluteCorrector','ak1PFResidualCorrector')
    )
ak1PFL1L2L3ResidualCorrectorChain = cms.Sequence(
    ak1PFL1OffsetCorrector * ak1PFL2RelativeCorrector * ak1PFL3AbsoluteCorrector * ak1PFResidualCorrector * ak1PFL1L2L3ResidualCorrector
)
ak1PFCHSL1L2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak1PFCHSL1OffsetCorrector','ak1PFCHSL2RelativeCorrector','ak1PFCHSL3AbsoluteCorrector','ak1PFCHSResidualCorrector')
    )
ak1PFCHSL1L2L3ResidualCorrectorChain = cms.Sequence(
    ak1PFCHSL1OffsetCorrector * ak1PFCHSL2RelativeCorrector * ak1PFCHSL3AbsoluteCorrector * ak1PFCHSResidualCorrector * ak1PFCHSL1L2L3ResidualCorrector
)
ak2PFL1L2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak2PFL1OffsetCorrector','ak2PFL2RelativeCorrector','ak2PFL3AbsoluteCorrector','ak2PFResidualCorrector')
    )
ak2PFL1L2L3ResidualCorrectorChain = cms.Sequence(
    ak2PFL1OffsetCorrector * ak2PFL2RelativeCorrector * ak2PFL3AbsoluteCorrector * ak2PFResidualCorrector * ak2PFL1L2L3ResidualCorrector
)
ak2PFCHSL1L2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak2PFCHSL1OffsetCorrector','ak2PFCHSL2RelativeCorrector','ak2PFCHSL3AbsoluteCorrector','ak2PFCHSResidualCorrector')
    )
ak2PFCHSL1L2L3ResidualCorrectorChain = cms.Sequence(
    ak2PFCHSL1OffsetCorrector * ak2PFCHSL2RelativeCorrector * ak2PFCHSL3AbsoluteCorrector * ak2PFCHSResidualCorrector * ak2PFCHSL1L2L3ResidualCorrector
)
ak3PFL1L2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak3PFL1OffsetCorrector','ak3PFL2RelativeCorrector','ak3PFL3AbsoluteCorrector','ak3PFResidualCorrector')
    )
ak3PFL1L2L3ResidualCorrectorChain = cms.Sequence(
    ak3PFL1OffsetCorrector * ak3PFL2RelativeCorrector * ak3PFL3AbsoluteCorrector * ak3PFResidualCorrector * ak3PFL1L2L3ResidualCorrector
)
ak3PFCHSL1L2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak3PFCHSL1OffsetCorrector','ak3PFCHSL2RelativeCorrector','ak3PFCHSL3AbsoluteCorrector','ak3PFCHSResidualCorrector')
    )
ak3PFCHSL1L2L3ResidualCorrectorChain = cms.Sequence(
    ak3PFCHSL1OffsetCorrector * ak3PFCHSL2RelativeCorrector * ak3PFCHSL3AbsoluteCorrector * ak3PFCHSResidualCorrector * ak3PFCHSL1L2L3ResidualCorrector
)
ak5PFL1L2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak5PFL1OffsetCorrector','ak5PFL2RelativeCorrector','ak5PFL3AbsoluteCorrector','ak5PFResidualCorrector')
    )
ak5PFL1L2L3ResidualCorrectorChain = cms.Sequence(
    ak5PFL1OffsetCorrector * ak5PFL2RelativeCorrector * ak5PFL3AbsoluteCorrector * ak5PFResidualCorrector * ak5PFL1L2L3ResidualCorrector
)
ak5PFCHSL1L2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak5PFCHSL1OffsetCorrector','ak5PFCHSL2RelativeCorrector','ak5PFCHSL3AbsoluteCorrector','ak5PFCHSResidualCorrector')
    )
ak5PFCHSL1L2L3ResidualCorrectorChain = cms.Sequence(
    ak5PFCHSL1OffsetCorrector * ak5PFCHSL2RelativeCorrector * ak5PFCHSL3AbsoluteCorrector * ak5PFCHSResidualCorrector * ak5PFCHSL1L2L3ResidualCorrector
)
ak6PFL1L2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak6PFL1OffsetCorrector','ak6PFL2RelativeCorrector','ak6PFL3AbsoluteCorrector','ak6PFResidualCorrector')
    )
ak6PFL1L2L3ResidualCorrectorChain = cms.Sequence(
    ak6PFL1OffsetCorrector * ak6PFL2RelativeCorrector * ak6PFL3AbsoluteCorrector * ak6PFResidualCorrector * ak6PFL1L2L3ResidualCorrector
)
ak6PFCHSL1L2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak6PFCHSL1OffsetCorrector','ak6PFCHSL2RelativeCorrector','ak6PFCHSL3AbsoluteCorrector','ak6PFCHSResidualCorrector')
    )
ak6PFCHSL1L2L3ResidualCorrectorChain = cms.Sequence(
    ak6PFCHSL1OffsetCorrector * ak6PFCHSL2RelativeCorrector * ak6PFCHSL3AbsoluteCorrector * ak6PFCHSResidualCorrector * ak6PFCHSL1L2L3ResidualCorrector
)
ak7PFL1L2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak7PFL1OffsetCorrector','ak7PFL2RelativeCorrector','ak7PFL3AbsoluteCorrector','ak7PFResidualCorrector')
    )
ak7PFL1L2L3ResidualCorrectorChain = cms.Sequence(
    ak7PFL1OffsetCorrector * ak7PFL2RelativeCorrector * ak7PFL3AbsoluteCorrector * ak7PFResidualCorrector * ak7PFL1L2L3ResidualCorrector
)
ak7PFCHSL1L2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak7PFCHSL1OffsetCorrector','ak7PFCHSL2RelativeCorrector','ak7PFCHSL3AbsoluteCorrector','ak7PFCHSResidualCorrector')
    )
ak7PFCHSL1L2L3ResidualCorrectorChain = cms.Sequence(
    ak7PFCHSL1OffsetCorrector * ak7PFCHSL2RelativeCorrector * ak7PFCHSL3AbsoluteCorrector * ak7PFCHSResidualCorrector * ak7PFCHSL1L2L3ResidualCorrector
)
ak8PFL1L2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak8PFL1OffsetCorrector','ak8PFL2RelativeCorrector','ak8PFL3AbsoluteCorrector','ak8PFResidualCorrector')
    )
ak8PFL1L2L3ResidualCorrectorChain = cms.Sequence(
    ak8PFL1OffsetCorrector * ak8PFL2RelativeCorrector * ak8PFL3AbsoluteCorrector * ak8PFResidualCorrector * ak8PFL1L2L3ResidualCorrector
)
ak8PFCHSL1L2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak8PFCHSL1OffsetCorrector','ak8PFCHSL2RelativeCorrector','ak8PFCHSL3AbsoluteCorrector','ak8PFCHSResidualCorrector')
    )
ak8PFCHSL1L2L3ResidualCorrectorChain = cms.Sequence(
    ak8PFCHSL1OffsetCorrector * ak8PFCHSL2RelativeCorrector * ak8PFCHSL3AbsoluteCorrector * ak8PFCHSResidualCorrector * ak8PFCHSL1L2L3ResidualCorrector
)
ak9PFL1L2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak9PFL1OffsetCorrector','ak9PFL2RelativeCorrector','ak9PFL3AbsoluteCorrector','ak9PFResidualCorrector')
    )
ak9PFL1L2L3ResidualCorrectorChain = cms.Sequence(
    ak9PFL1OffsetCorrector * ak9PFL2RelativeCorrector * ak9PFL3AbsoluteCorrector * ak9PFResidualCorrector * ak9PFL1L2L3ResidualCorrector
)
ak9PFCHSL1L2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak9PFCHSL1OffsetCorrector','ak9PFCHSL2RelativeCorrector','ak9PFCHSL3AbsoluteCorrector','ak9PFCHSResidualCorrector')
    )
ak9PFCHSL1L2L3ResidualCorrectorChain = cms.Sequence(
    ak9PFCHSL1OffsetCorrector * ak9PFCHSL2RelativeCorrector * ak9PFCHSL3AbsoluteCorrector * ak9PFCHSResidualCorrector * ak9PFCHSL1L2L3ResidualCorrector
)
ak10PFL1L2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak10PFL1OffsetCorrector','ak10PFL2RelativeCorrector','ak10PFL3AbsoluteCorrector','ak10PFResidualCorrector')
    )
ak10PFL1L2L3ResidualCorrectorChain = cms.Sequence(
    ak10PFL1OffsetCorrector * ak10PFL2RelativeCorrector * ak10PFL3AbsoluteCorrector * ak10PFResidualCorrector * ak10PFL1L2L3ResidualCorrector
)
ak10PFCHSL1L2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak10PFCHSL1OffsetCorrector','ak10PFCHSL2RelativeCorrector','ak10PFCHSL3AbsoluteCorrector','ak10PFCHSResidualCorrector')
    )
ak10PFCHSL1L2L3ResidualCorrectorChain = cms.Sequence(
    ak10PFCHSL1OffsetCorrector * ak10PFCHSL2RelativeCorrector * ak10PFCHSL3AbsoluteCorrector * ak10PFCHSResidualCorrector * ak10PFCHSL1L2L3ResidualCorrector
)

kt4PFL1L2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('kt4PFL1OffsetCorrector','kt4PFL2RelativeCorrector','kt4PFL3AbsoluteCorrector','kt4PFResidualCorrector')
    )
kt4PFL1L2L3ResidualCorrectorChain = cms.Sequence(
    kt4PFL1OffsetCorrector * kt4PFL2RelativeCorrector * kt4PFL3AbsoluteCorrector * kt4PFResidualCorrector * kt4PFL1L2L3ResidualCorrector
)
kt6PFL1L2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('kt6PFL1OffsetCorrector','kt6PFL2RelativeCorrector','kt6PFL3AbsoluteCorrector','kt6PFResidualCorrector')
    )
kt6PFL1L2L3ResidualCorrectorChain = cms.Sequence(
    kt6PFL1OffsetCorrector * kt6PFL2RelativeCorrector * kt6PFL3AbsoluteCorrector * kt6PFResidualCorrector * kt6PFL1L2L3ResidualCorrector
)
ic5PFL1L2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ic5PFL1OffsetCorrector','ic5PFL2RelativeCorrector','ic5PFL3AbsoluteCorrector','ic5PFResidualCorrector')
    )
ic5PFL1L2L3ResidualCorrectorChain = cms.Sequence(
    ic5PFL1OffsetCorrector * ic5PFL2RelativeCorrector * ic5PFL3AbsoluteCorrector * ic5PFResidualCorrector * ic5PFL1L2L3ResidualCorrector
)
#--- JPT needs the L1JPTOffset to account for the ZSP changes.
#--- L1JPTOffset is NOT the same as L1Offset !!!!!
ak7JPTL1L2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak7JPTL1OffsetCorrector','ak7L1JPTOffsetCorrector','ak7JPTL2RelativeCorrector','ak7JPTL3AbsoluteCorrector','ak7JPTResidualCorrector')
    )
ak7JPTL1L2L3ResidualCorrectorChain = cms.Sequence(
    ak7JPTL1OffsetCorrector * ak7L1JPTOffsetCorrectorChain * ak7JPTL2RelativeCorrector * ak7JPTL3AbsoluteCorrector * ak7JPTResidualCorrector * ak7JPTL1L2L3ResidualCorrector
)

# L1FastL2L3 CORRECTORS
ak7CaloL1FastL2L3Corrector = ak7CaloL2L3Corrector.clone()
ak7CaloL1FastL2L3Corrector.correctors.insert(0,'ak4CaloL1FastjetCorrector')
ak7CaloL1FastL2L3CorrectorChain = cms.Sequence(
    ak4CaloL1FastjetCorrector * ak7CaloL2RelativeCorrector * ak7CaloL3AbsoluteCorrector * ak7CaloL1FastL2L3Corrector
)
kt4CaloL1FastL2L3Corrector = kt4CaloL2L3Corrector.clone()
kt4CaloL1FastL2L3Corrector.correctors.insert(0,'ak4CaloL1FastjetCorrector')
kt4CaloL1FastL2L3CorrectorChain = cms.Sequence(
    ak4CaloL1FastjetCorrector * kt4CaloL2RelativeCorrector * kt4CaloL3AbsoluteCorrector * kt4CaloL1FastL2L3Corrector
)
kt6CaloL1FastL2L3Corrector = kt6CaloL2L3Corrector.clone()
kt6CaloL1FastL2L3Corrector.correctors.insert(0,'ak4CaloL1FastjetCorrector')
kt6CaloL1FastL2L3CorrectorChain = cms.Sequence(
    ak4CaloL1FastjetCorrector * kt6CaloL2RelativeCorrector * kt6CaloL3AbsoluteCorrector * kt6CaloL1FastL2L3Corrector
)
ic5CaloL1FastL2L3Corrector = ic5CaloL2L3Corrector.clone()
ic5CaloL1FastL2L3Corrector.correctors.insert(0,'ak4CaloL1FastjetCorrector')
ic5CaloL1FastL2L3CorrectorChain = cms.Sequence(
    ak4CaloL1FastjetCorrector * ic5CaloL2RelativeCorrector * ic5CaloL3AbsoluteCorrector * ic5CaloL1FastL2L3Corrector
)

ak7PFL1FastL2L3Corrector = ak7PFL2L3Corrector.clone()
ak7PFL1FastL2L3Corrector.correctors.insert(0,'ak4PFL1FastjetCorrector')
ak7PFL1FastL2L3CorrectorChain = cms.Sequence(
    ak4PFL1FastjetCorrector * ak7PFL2RelativeCorrector * ak7PFL3AbsoluteCorrector * ak7PFL1FastL2L3Corrector
)
ak7PFCHSL1FastL2L3Corrector = ak7PFCHSL2L3Corrector.clone()
ak7PFCHSL1FastL2L3Corrector.correctors.insert(0,'ak4PFCHSL1FastjetCorrector')
ak7PFCHSL1FastL2L3CorrectorChain = cms.Sequence(
    ak4PFCHSL1FastjetCorrector * ak7PFCHSL2RelativeCorrector * ak7PFCHSL3AbsoluteCorrector * ak7PFCHSL1FastL2L3Corrector
)
kt4PFL1FastL2L3Corrector = kt4PFL2L3Corrector.clone()
kt4PFL1FastL2L3Corrector.correctors.insert(0,'ak4PFL1FastjetCorrector')
kt4PFL1FastL2L3CorrectorChain = cms.Sequence(
    ak4PFL1FastjetCorrector * kt4PFL2RelativeCorrector * kt4PFL3AbsoluteCorrector * kt4PFL1FastL2L3Corrector
)
kt6PFL1FastL2L3Corrector = kt6PFL2L3Corrector.clone()
kt6PFL1FastL2L3Corrector.correctors.insert(0,'ak4PFL1FastjetCorrector')
kt6PFL1FastL2L3CorrectorChain = cms.Sequence(
    ak4PFL1FastjetCorrector * kt6PFL2RelativeCorrector * kt6PFL3AbsoluteCorrector * kt6PFL1FastL2L3Corrector
)
ic5PFL1FastL2L3Corrector = ic5PFL2L3Corrector.clone()
ic5PFL1FastL2L3Corrector.correctors.insert(0,'ak4PFL1FastjetCorrector')
ic5PFL1FastL2L3CorrectorChain = cms.Sequence(
    ak4PFL1FastjetCorrector * ic5PFL2RelativeCorrector * ic5PFL3AbsoluteCorrector * ic5PFL1FastL2L3Corrector
)

ak4TrackL1FastL2L3Corrector = ak4TrackL2L3Corrector.clone()
ak4TrackL1FastL2L3Corrector.correctors.insert(0,'ak4CaloL1FastjetCorrector')
ak4TrackL1FastL2L3CorrectorChain = cms.Sequence(
    ak4CaloL1FastjetCorrector * ak4TrackL2RelativeCorrector * ak4TrackL3AbsoluteCorrector * ak4TrackL1FastL2L3Corrector
)

# L1FastL2L3Residual CORRECTORS
ak7CaloL1FastL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak7CaloL1FastjetCorrector','ak7CaloL2RelativeCorrector','ak7CaloL3AbsoluteCorrector','ak7CaloResidualCorrector')
    )
ak7CaloL1FastL2L3ResidualCorrectorChain = cms.Sequence(
    ak7CaloL1FastjetCorrector * ak7CaloL2RelativeCorrector * ak7CaloL3AbsoluteCorrector * ak7CaloResidualCorrector * ak7CaloL1FastL2L3ResidualCorrector
)
kt4CaloL1FastL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('kt4CaloL1FastjetCorrector','kt4CaloL2RelativeCorrector','kt4CaloL3AbsoluteCorrector','kt4CaloResidualCorrector')
    )
kt4CaloL1FastL2L3ResidualCorrectorChain = cms.Sequence(
    kt4CaloL1FastjetCorrector * kt4CaloL2RelativeCorrector * kt4CaloL3AbsoluteCorrector * kt4CaloResidualCorrector * kt4CaloL1FastL2L3ResidualCorrector
)
kt6CaloL1FastL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('kt6CaloL1FastjetCorrector','kt6CaloL2RelativeCorrector','kt6CaloL3AbsoluteCorrector','kt6CaloResidualCorrector')
    )
kt6CaloL1FastL2L3ResidualCorrectorChain = cms.Sequence(
    kt6CaloL1FastjetCorrector * kt6CaloL2RelativeCorrector * kt6CaloL3AbsoluteCorrector * kt6CaloResidualCorrector * kt6CaloL1FastL2L3ResidualCorrector
)
ic5CaloL1FastL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ic5CaloL1FastjetCorrector','ic5CaloL2RelativeCorrector','ic5CaloL3AbsoluteCorrector','ic5CaloResidualCorrector')
    )
ic5CaloL1FastL2L3ResidualCorrectorChain = cms.Sequence(
    ic5CaloL1FastjetCorrector * ic5CaloL2RelativeCorrector * ic5CaloL3AbsoluteCorrector * ic5CaloResidualCorrector * ic5CaloL1FastL2L3ResidualCorrector
)



ak1PFL1FastjetL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak1PFL1FastjetCorrector','ak1PFL2RelativeCorrector','ak1PFL3AbsoluteCorrector','ak1PFResidualCorrector')
    )
ak1PFL1FastjetL2L3ResidualCorrectorChain = cms.Sequence(
    ak1PFL1FastjetCorrector * ak1PFL2RelativeCorrector * ak1PFL3AbsoluteCorrector * ak1PFResidualCorrector * ak1PFL1FastjetL2L3ResidualCorrector
)
ak1PFCHSL1FastjetL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak1PFCHSL1FastjetCorrector','ak1PFCHSL2RelativeCorrector','ak1PFCHSL3AbsoluteCorrector','ak1PFCHSResidualCorrector')
    )
ak1PFCHSL1FastjetL2L3ResidualCorrectorChain = cms.Sequence(
    ak1PFCHSL1FastjetCorrector * ak1PFCHSL2RelativeCorrector * ak1PFCHSL3AbsoluteCorrector * ak1PFCHSResidualCorrector * ak1PFCHSL1FastjetL2L3ResidualCorrector
)
ak2PFL1FastjetL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak2PFL1FastjetCorrector','ak2PFL2RelativeCorrector','ak2PFL3AbsoluteCorrector','ak2PFResidualCorrector')
    )
ak2PFL1FastjetL2L3ResidualCorrectorChain = cms.Sequence(
    ak2PFL1FastjetCorrector * ak2PFL2RelativeCorrector * ak2PFL3AbsoluteCorrector * ak2PFResidualCorrector * ak2PFL1FastjetL2L3ResidualCorrector
)
ak2PFCHSL1FastjetL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak2PFCHSL1FastjetCorrector','ak2PFCHSL2RelativeCorrector','ak2PFCHSL3AbsoluteCorrector','ak2PFCHSResidualCorrector')
    )
ak2PFCHSL1FastjetL2L3ResidualCorrectorChain = cms.Sequence(
    ak2PFCHSL1FastjetCorrector * ak2PFCHSL2RelativeCorrector * ak2PFCHSL3AbsoluteCorrector * ak2PFCHSResidualCorrector * ak2PFCHSL1FastjetL2L3ResidualCorrector
)
ak3PFL1FastjetL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak3PFL1FastjetCorrector','ak3PFL2RelativeCorrector','ak3PFL3AbsoluteCorrector','ak3PFResidualCorrector')
    )
ak3PFL1FastjetL2L3ResidualCorrectorChain = cms.Sequence(
    ak3PFL1FastjetCorrector * ak3PFL2RelativeCorrector * ak3PFL3AbsoluteCorrector * ak3PFResidualCorrector * ak3PFL1FastjetL2L3ResidualCorrector
)
ak3PFCHSL1FastjetL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak3PFCHSL1FastjetCorrector','ak3PFCHSL2RelativeCorrector','ak3PFCHSL3AbsoluteCorrector','ak3PFCHSResidualCorrector')
    )
ak3PFCHSL1FastjetL2L3ResidualCorrectorChain = cms.Sequence(
    ak3PFCHSL1FastjetCorrector * ak3PFCHSL2RelativeCorrector * ak3PFCHSL3AbsoluteCorrector * ak3PFCHSResidualCorrector * ak3PFCHSL1FastjetL2L3ResidualCorrector
)
ak5PFL1FastjetL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak5PFL1FastjetCorrector','ak5PFL2RelativeCorrector','ak5PFL3AbsoluteCorrector','ak5PFResidualCorrector')
    )
ak5PFL1FastjetL2L3ResidualCorrectorChain = cms.Sequence(
    ak5PFL1FastjetCorrector * ak5PFL2RelativeCorrector * ak5PFL3AbsoluteCorrector * ak5PFResidualCorrector *  ak5PFL1FastjetL2L3ResidualCorrector
)
ak5PFCHSL1FastjetL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak5PFCHSL1FastjetCorrector','ak5PFCHSL2RelativeCorrector','ak5PFCHSL3AbsoluteCorrector','ak5PFCHSResidualCorrector')
    )
ak5PFCHSL1FastjetL2L3ResidualCorrectorChain = cms.Sequence(
    ak5PFCHSL1FastjetCorrector * ak5PFCHSL2RelativeCorrector * ak5PFCHSL3AbsoluteCorrector * ak5PFCHSResidualCorrector * ak5PFCHSL1FastjetL2L3ResidualCorrector
)
ak6PFL1FastjetL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak6PFL1FastjetCorrector','ak6PFL2RelativeCorrector','ak6PFL3AbsoluteCorrector','ak6PFResidualCorrector')
    )
ak6PFL1FastjetL2L3ResidualCorrectorChain = cms.Sequence(
    ak6PFL1FastjetCorrector * ak6PFL2RelativeCorrector * ak6PFL3AbsoluteCorrector * ak6PFResidualCorrector *  ak6PFL1FastjetL2L3ResidualCorrector
)
ak6PFCHSL1FastjetL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak6PFCHSL1FastjetCorrector','ak6PFCHSL2RelativeCorrector','ak6PFCHSL3AbsoluteCorrector','ak6PFCHSResidualCorrector')
    )
ak6PFCHSL1FastjetL2L3ResidualCorrectorChain = cms.Sequence(
    ak6PFCHSL1FastjetCorrector * ak6PFCHSL2RelativeCorrector * ak6PFCHSL3AbsoluteCorrector * ak6PFCHSResidualCorrector * ak6PFCHSL1FastjetL2L3ResidualCorrector
)
ak7PFL1FastjetL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak7PFL1FastjetCorrector','ak7PFL2RelativeCorrector','ak7PFL3AbsoluteCorrector','ak7PFResidualCorrector')
    )
ak7PFL1FastjetL2L3ResidualCorrectorChain = cms.Sequence(
    ak7PFL1FastjetCorrector * ak7PFL2RelativeCorrector * ak7PFL3AbsoluteCorrector * ak7PFResidualCorrector * ak7PFL1FastjetL2L3ResidualCorrector
)
ak7PFCHSL1FastjetL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak7PFCHSL1FastjetCorrector','ak7PFCHSL2RelativeCorrector','ak7PFCHSL3AbsoluteCorrector','ak7PFCHSResidualCorrector')
    )
ak7PFCHSL1FastjetL2L3ResidualCorrectorChain = cms.Sequence(
    ak7PFCHSL1FastjetCorrector * ak7PFCHSL2RelativeCorrector * ak7PFCHSL3AbsoluteCorrector * ak7PFCHSResidualCorrector * ak7PFCHSL1FastjetL2L3ResidualCorrector
)
ak8PFL1FastjetL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak8PFL1FastjetCorrector','ak8PFL2RelativeCorrector','ak8PFL3AbsoluteCorrector','ak8PFResidualCorrector')
    )
ak8PFL1FastjetL2L3ResidualCorrectorChain = cms.Sequence(
    ak8PFL1FastjetCorrector * ak8PFL2RelativeCorrector * ak8PFL3AbsoluteCorrector * ak8PFResidualCorrector * ak8PFL1FastjetL2L3ResidualCorrector
)
ak8PFCHSL1FastjetL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak8PFCHSL1FastjetCorrector','ak8PFCHSL2RelativeCorrector','ak8PFCHSL3AbsoluteCorrector','ak8PFCHSResidualCorrector')
    )
ak8PFCHSL1FastjetL2L3ResidualCorrectorChain = cms.Sequence(
    ak8PFCHSL1FastjetCorrector * ak8PFCHSL2RelativeCorrector * ak8PFCHSL3AbsoluteCorrector * ak8PFCHSResidualCorrector * ak8PFCHSL1FastjetL2L3ResidualCorrector
)
ak9PFL1FastjetL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak9PFL1FastjetCorrector','ak9PFL2RelativeCorrector','ak9PFL3AbsoluteCorrector','ak9PFResidualCorrector')
    )
ak9PFL1FastjetL2L3ResidualCorrectorChain = cms.Sequence(
    ak9PFL1FastjetCorrector * ak9PFL2RelativeCorrector * ak9PFL3AbsoluteCorrector * ak9PFResidualCorrector * ak9PFL1FastjetL2L3ResidualCorrector
)
ak9PFCHSL1FastjetL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak9PFCHSL1FastjetCorrector','ak9PFCHSL2RelativeCorrector','ak9PFCHSL3AbsoluteCorrector','ak9PFCHSResidualCorrector')
    )
ak9PFCHSL1FastjetL2L3ResidualCorrectorChain = cms.Sequence(
    ak9PFCHSL1FastjetCorrector * ak9PFCHSL2RelativeCorrector * ak9PFCHSL3AbsoluteCorrector * ak9PFCHSResidualCorrector * ak9PFCHSL1FastjetL2L3ResidualCorrector
)
ak10PFL1FastjetL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak10PFL1FastjetCorrector','ak10PFL2RelativeCorrector','ak10PFL3AbsoluteCorrector','ak10PFResidualCorrector')
    )
ak10PFL1FastjetL2L3ResidualCorrectorChain = cms.Sequence(
    ak10PFL1FastjetCorrector * ak10PFL2RelativeCorrector * ak10PFL3AbsoluteCorrector * ak10PFResidualCorrector * ak10PFL1FastjetL2L3ResidualCorrector
)
ak10PFCHSL1FastjetL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak10PFCHSL1FastjetCorrector','ak10PFCHSL2RelativeCorrector','ak10PFCHSL3AbsoluteCorrector','ak10PFCHSResidualCorrector')
    )
ak10PFCHSL1FastjetL2L3ResidualCorrectorChain = cms.Sequence(
    ak10PFCHSL1FastjetCorrector * ak10PFCHSL2RelativeCorrector * ak10PFCHSL3AbsoluteCorrector * ak10PFCHSResidualCorrector * ak10PFCHSL1FastjetL2L3ResidualCorrector
)

kt4PFL1FastL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('kt4PFL1FastjetCorrector','kt4PFL2RelativeCorrector','kt4PFL3AbsoluteCorrector','kt4PFResidualCorrector')
    )
kt4PFL1FastL2L3ResidualCorrectorChain = cms.Sequence(
    kt4PFL1FastjetCorrector * kt4PFL2RelativeCorrector * kt4PFL3AbsoluteCorrector * kt4PFResidualCorrector * kt4PFL1FastL2L3ResidualCorrector
)
kt6PFL1FastL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('kt6PFL1FastjetCorrector','kt6PFL2RelativeCorrector','kt6PFL3AbsoluteCorrector','kt6PFResidualCorrector')
    )
kt6PFL1FastL2L3ResidualCorrectorChain = cms.Sequence(
    kt6PFL1FastjetCorrector * kt6PFL2RelativeCorrector * kt6PFL3AbsoluteCorrector * kt6PFResidualCorrector * kt6PFL1FastL2L3ResidualCorrector
)
ic5PFL1FastL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ic5PFL1FastjetCorrector','ic5PFL2RelativeCorrector','ic5PFL3AbsoluteCorrector','ic5PFResidualCorrector')
    )
ic5PFL1FastL2L3ResidualCorrectorChain = cms.Sequence(
    ic5PFL1FastjetCorrector * ic5PFL2RelativeCorrector * ic5PFL3AbsoluteCorrector * ic5PFResidualCorrector * ic5PFL1FastL2L3ResidualCorrector
)
#--- JPT needs the L1JPTOffset to account for the ZSP changes.
#--- L1JPTOffset is NOT the same as L1Offset !!!!!
ak7JPTL1FastL2L3ResidualCorrector = cms.EDProducer(
    'ChainedJetCorrectorProducer',
    correctors = cms.VInputTag('ak7JPTL1FastjetCorrector','ak7L1JPTOffsetCorrector','ak7JPTL2RelativeCorrector','ak7JPTL3AbsoluteCorrector','ak7JPTResidualCorrector')
    )
ak7JPTL1FastL2L3ResidualCorrectorChain = cms.Sequence(
    ak7JPTL1FastjetCorrector * ak7L1JPTOffsetCorrectorChain * ak7JPTL2RelativeCorrector * ak7JPTL3AbsoluteCorrector * ak7JPTResidualCorrector *  ak7JPTL1FastL2L3ResidualCorrector
)

# L2L3L6 CORRECTORS
ak7CaloL2L3L6Corrector = ak7CaloL2L3Corrector.clone()
ak7CaloL2L3L6Corrector.correctors.append('ak7CaloL6SLBCorrector')
ak7CaloL2L3L6CorrectorChain = cms.Sequence(
    ak7CaloL2L3Corrector * ak7CaloL6SLBCorrector * ak7CaloL2L3L6Corrector
)
kt4CaloL2L3L6Corrector = kt4CaloL2L3Corrector.clone()
kt4CaloL2L3L6Corrector.correctors.append('kt4CaloL6SLBCorrector')
kt4CaloL2L3L6CorrectorChain = cms.Sequence(
    ak7CaloL2L3Corrector * kt4CaloL6SLBCorrector * kt4CaloL2L3L6Corrector
)
kt6CaloL2L3L6Corrector = kt6CaloL2L3Corrector.clone()
kt6CaloL2L3L6Corrector.correctors.append('kt6CaloL6SLBCorrector')
kt6CaloL2L3L6CorrectorChain = cms.Sequence(
    ak7CaloL2L3Corrector * kt6CaloL6SLBCorrector * kt6CaloL2L3L6Corrector
)
ic5CaloL2L3L6Corrector = ic5CaloL2L3Corrector.clone()
ic5CaloL2L3L6Corrector.correctors.append('ic5CaloL6SLBCorrector')
ic5CaloL2L3L6CorrectorChain = cms.Sequence(
    ak7CaloL2L3Corrector * ic5CaloL6SLBCorrector * ic5CaloL2L3L6Corrector
)

ak7PFL2L3L6Corrector = ak7PFL2L3Corrector.clone()
ak7PFL2L3L6Corrector.correctors.append('ak7PFL6SLBCorrector')
ak7PFL2L3L6CorrectorChain = cms.Sequence(
    ak7PFL2L3Corrector * ak7PFL6SLBCorrector * ak7PFL2L3L6Corrector
)
kt4PFL2L3L6Corrector = kt4PFL2L3Corrector.clone()
kt4PFL2L3L6Corrector.correctors.append('kt4PFL6SLBCorrector')
Chain = cms.Sequence(
    kt4PFL2L3Corrector * kt4PFL6SLBCorrector * kt4PFL2L3L6Corrector
)
kt6PFL2L3L6Corrector = kt6PFL2L3Corrector.clone()
kt6PFL2L3L6Corrector.correctors.append('kt6PFL6SLBCorrector')
kt6PFL2L3L6CorrectorChain = cms.Sequence(
    kt4PFL2L3Corrector * kt6PFL6SLBCorrector * kt6PFL2L3L6Corrector
)
ic5PFL2L3L6Corrector = ic5PFL2L3Corrector.clone()
ic5PFL2L3L6Corrector.correctors.append('ic5PFL6SLBCorrector')
ic5PFL2L3L6CorrectorChain = cms.Sequence(
    kt4PFL2L3Corrector * ic5PFL6SLBCorrector * ic5PFL2L3L6Corrector
)


# L1L2L3L6 CORRECTORS
ak7CaloL1FastL2L3L6Corrector = ak7CaloL1L2L3Corrector.clone()
ak7CaloL1FastL2L3L6Corrector.correctors.append('ak7CaloL6SLBCorrector')
ak7CaloL1FastL2L3L6CorrectorChain = cms.Sequence(
    ak7CaloL1L2L3Corrector * ak7CaloL6SLBCorrector * ak7CaloL1FastL2L3L6Corrector
)
kt4CaloL1FastL2L3L6Corrector = kt4CaloL1L2L3Corrector.clone()
kt4CaloL1FastL2L3L6Corrector.correctors.append('kt4CaloL6SLBCorrector')
kt4CaloL1FastL2L3L6CorrectorChain = cms.Sequence(
    ak7CaloL1L2L3Corrector * kt4CaloL6SLBCorrector * kt4CaloL1FastL2L3L6Corrector
)
kt6CaloL1FastL2L3L6Corrector = kt6CaloL1L2L3Corrector.clone()
kt6CaloL1FastL2L3L6Corrector.correctors.append('kt6CaloL6SLBCorrector')
kt6CaloL1FastL2L3L6CorrectorChain = cms.Sequence(
    ak7CaloL1L2L3Corrector * kt6CaloL6SLBCorrector * kt6CaloL1FastL2L3L6Corrector
)
ic5CaloL1FastL2L3L6Corrector = ic5CaloL1L2L3Corrector.clone()
ic5CaloL1FastL2L3L6Corrector.correctors.append('ic5CaloL6SLBCorrector')
ic5CaloL1FastL2L3L6CorrectorChain = cms.Sequence(
    ak7CaloL1L2L3Corrector * ic5CaloL6SLBCorrector * ic5CaloL1FastL2L3L6Corrector
)

ak7PFL1FastL2L3L6Corrector = ak7PFL1FastL2L3Corrector.clone()
ak7PFL1FastL2L3L6Corrector.correctors.append('ak7PFL6SLBCorrector')
ak7PFL1FastL2L3L6CorrectorChain = cms.Sequence(
    ak7CaloL1L2L3Corrector * ak7PFL6SLBCorrector * ak7PFL1FastL2L3L6Corrector
)
kt4PFL1FastL2L3L6Corrector = kt4PFL1FastL2L3Corrector.clone()
kt4PFL1FastL2L3L6Corrector.correctors.append('kt4PFL6SLBCorrector')
kt4PFL1FastL2L3L6CorrectorChain = cms.Sequence(
    kt4PFL1FastL2L3Corrector * kt4PFL6SLBCorrector * kt4PFL1FastL2L3L6Corrector
)
kt6PFL1FastL2L3L6Corrector = kt6PFL1FastL2L3Corrector.clone()
kt6PFL1FastL2L3L6Corrector.correctors.append('kt6PFL6SLBCorrector')
kt6PFL1FastL2L3L6CorrectorChain = cms.Sequence(
    kt4PFL1FastL2L3Corrector * kt6PFL6SLBCorrector * kt6PFL1FastL2L3L6Corrector
)
ic5PFL1FastL2L3L6Corrector = ic5PFL1FastL2L3Corrector.clone()
ic5PFL1FastL2L3L6Corrector.correctors.append('ic5PFL6SLBCorrector')
ic5PFL1FastL2L3L6CorrectorChain = cms.Sequence(
    kt4PFL1FastL2L3Corrector * ic5PFL6SLBCorrector * ic5PFL1FastL2L3L6Corrector
)
