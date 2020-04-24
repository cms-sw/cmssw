import FWCore.ParameterSet.Config as cms
from DQMOffline.RecoB.cTagCombinedSVVariables_cff import *

# CAT1 -> RecoVertex category with all soft lepton categories (NoSoftLepton, SoftMuon, SoftElec).
# CAT2 -> PseudoVertex category with all soft lepton categories (NoSoftLepton, SoftMuon, SoftElec).
# CAT3 -> NoVertex category with all soft lepton categories (NoSoftLepton, SoftMuon, SoftElec).
# CAT4 -> All vertex categories (RecoVertex, PseudoVertex, NoVertex) with SoftMuon, SoftElec categories.  

# combinedSecondaryVertex jet tag computer configuration
cTagCombinedSVAnalysisBlock = cms.PSet(
    parameters = cms.PSet(
        categoryVariable = cms.string('vertexLeptonCategory'),
	categories = cms.VPSet(cms.PSet(
            combinedSVAllVertexAllSoftLeptonCtagLVariables,
	    combinedSVPseudoVertexAllSoftLeptonCtagLVariables,
	    combinedSVRecoPseudoVertexAllSoftLeptonCtagLVariables,
            combinedSVRecoVertexAllSoftLeptonCtagLVariables,
            combinedSVAllVertexSoftLeptonCtagLVariables
	),
	   cms.PSet(
	       combinedSVAllVertexAllSoftLeptonCtagLVariables,
	       combinedSVRecoPseudoVertexAllSoftLeptonCtagLVariables,
               combinedSVRecoVertexAllSoftLeptonCtagLVariables
	   ),
	   cms.PSet(
	       combinedSVAllVertexAllSoftLeptonCtagLVariables,
	       combinedSVRecoPseudoVertexAllSoftLeptonCtagLVariables,
	       combinedSVPseudoVertexAllSoftLeptonCtagLVariables
           ),
	   cms.PSet(
	       combinedSVAllVertexAllSoftLeptonCtagLVariables
           ),
	   cms.PSet(
	       combinedSVAllVertexSoftLeptonCtagLVariables
           ))
    )
)
