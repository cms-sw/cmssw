import FWCore.ParameterSet.Config as cms


from RecoTracker.DeDx.dedxDiscriminator_Prod_cfi import *

dedxDiscrimBTag         = dedxDiscrimProd.clone()
dedxDiscrimBTag.Formula = cms.untracked.uint32(1)

dedxDiscrimSmi         = dedxDiscrimProd.clone()
dedxDiscrimSmi.Formula = cms.untracked.uint32(2)

dedxDiscrimASmi         = dedxDiscrimProd.clone()
dedxDiscrimASmi.Formula = cms.untracked.uint32(3)

doAlldEdXDiscriminators = cms.Sequence(dedxDiscrimProd * dedxDiscrimBTag * dedxDiscrimSmi * dedxDiscrimASmi)
