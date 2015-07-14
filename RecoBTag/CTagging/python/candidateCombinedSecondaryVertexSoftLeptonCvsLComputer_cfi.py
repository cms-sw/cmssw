import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.combinedSecondaryVertexCommon_cff import *

candidateCombinedSecondaryVertexSoftLeptonCvsLComputer = cms.ESProducer(
   "CandidateCombinedSecondaryVertexSoftLeptonCtagLESProducer",
   combinedSecondaryVertexCommon,
   useCategories = cms.bool(True),
   calibrationRecords = cms.vstring(
      'CombinedSVRecoVertexNoSoftLeptonCvsL', 
      'CombinedSVPseudoVertexNoSoftLeptonCvsL', 
      'CombinedSVNoVertexNoSoftLeptonCvsL',
      'CombinedSVRecoVertexSoftMuonCvsL', 
      'CombinedSVPseudoVertexSoftMuonCvsL', 
      'CombinedSVNoVertexSoftMuonCvsL',
      'CombinedSVRecoVertexSoftElectronCvsL', 
      'CombinedSVPseudoVertexSoftElectronCvsL', 
      'CombinedSVNoVertexSoftElectronCvsL'),
   categoryVariableName = cms.string('vertexLeptonCategory')
)

