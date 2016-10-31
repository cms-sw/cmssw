import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.combinedSecondaryVertexCommon_cff import *

candidateCombinedSecondaryVertexSoftLeptonCvsLComputer = cms.ESProducer(
   "CandidateCombinedSecondaryVertexSoftLeptonCvsLESProducer",
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
   recordLabel = cms.string(''),
   categoryVariableName = cms.string('vertexLeptonCategory')
)
