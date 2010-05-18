import FWCore.ParameterSet.Config as cms

L2RelativeCorrectionService = cms.ESSource('LXXXCorrectionService',
  era = cms.string('Summer09_7TeV_ReReco332'),
  level     = cms.string('L2Relative'),
  algorithm = cms.string('IC5Calo'),
  section   = cms.string('') 
)

L3AbsoluteCorrectionService = cms.ESSource('LXXXCorrectionService',
   era = cms.string('Summer09_7TeV_ReReco332'),
   level     = cms.string('L3Absolute'),
   algorithm = cms.string('IC5Calo'),
   section   = cms.string('')
)
 
MCJetCorrectorIcone5 = cms.ESSource( "JetCorrectionServiceChain",
   label = cms.string( "MCJetCorrectorIcone5" ),
   appendToDataLabel = cms.string( "" ),
   correctors = cms.vstring( 'L2RelativeCorrectionService',
     'L3AbsoluteCorrectionService' )
)

MCJetCorrectorIcone5HF07 = cms.ESSource('LXXXCorrectionService',
   era = cms.string('HLT'),
   level     = cms.string('L2Relative'),
   algorithm = cms.string(''),
   section   = cms.string('')
)

MCJetCorrectorIcone5Unit = cms.ESSource('LXXXCorrectionService',
   era = cms.string('HLT'),
   level     = cms.string('L2RelativeFlat'),
   algorithm = cms.string(''),
   section   = cms.string('')
)

