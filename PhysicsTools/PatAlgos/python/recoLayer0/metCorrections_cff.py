import FWCore.ParameterSet.Config as cms

#from PhysicsTools.PatAlgos.recoLayer0.jetCorrFactors_cfi import *
from JetMETCorrections.Type1MET.pfMETCorrections_cff import *
from JetMETCorrections.Type1MET.caloMETCorrections_cff import *
#from JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff import *

pfJetMETcorrCHS          = pfJetMETcorr.clone( src = cms.InputTag( "ak4PFJetsCHS" ) )
pfType1CorrectedMetCHS   = pfType1CorrectedMet.clone( srcType1Corrections = cms.VInputTag( cms.InputTag( "pfJetMETcorrCHS", "type1" ) ) )
pfType1p2CorrectedMetCHS = pfType1p2CorrectedMet.clone( srcType1Corrections = cms.VInputTag( cms.InputTag( "pfJetMETcorrCHS", "type1" ) ) )

## for scheduled mode
patMETCorrections = cms.Sequence(produceCaloMETCorrections+producePFMETCorrections)
