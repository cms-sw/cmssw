import FWCore.ParameterSet.Config as cms


bTagPostValidation = cms.EDAnalyzer("HLTBTagHarvestingAnalyzer",
   HLTPathNames     = cms.vstring('HLT_DiJet40Eta2p6_BTagIP3DFastPV'),
   minTags=cms.vdouble(3.41), # TCHP , 6 -- TCH6
   maxTag=cms.double(100.),
  # MC stuff
   mcFlavours = cms.PSet(
      light = cms.vuint32(1, 2, 3, 21),   # udsg
      c = cms.vuint32(4),
      b = cms.vuint32(5),
      g = cms.vuint32(21),
      uds = cms.vuint32(1, 2, 3)
    )

)
