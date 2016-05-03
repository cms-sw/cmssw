import FWCore.ParameterSet.Config as cms
#use import as to mask it to process.load() 
import RecoBTag.SecondaryVertex.candidateCombinedSecondaryVertexSoftLeptonComputer_cfi as sl_cfg 
from RecoBTag.CTagging.training_settings import c_vs_l_vars_vpset, c_vs_b_vars_vpset
from RecoBTag.SecondaryVertex.combinedSecondaryVertexCommon_cff import *

charmTagsComputerCvsL = cms.ESProducer(
   'CharmTaggerESProducer',
   combinedSecondaryVertexCommon,
   #clone the cfg only
   slComputerCfg = cms.PSet(
      **sl_cfg.candidateCombinedSecondaryVertexSoftLeptonComputer.parameters_()
      ),
   weightFile = cms.FileInPath('RecoBTag/CTagging/data/c_vs_udsg_sklearn.weight.xml'),
   variables = c_vs_l_vars_vpset,
   computer = cms.ESInputTag('combinedSecondaryVertexSoftLeptonComputer'),
   tagInfos = cms.VInputTag(
      cms.InputTag('pfImpactParameterTagInfos'),
      cms.InputTag('pfInclusiveSecondaryVertexFinderCvsLTagInfos'),
      cms.InputTag('softPFMuonsTagInfos'),
      cms.InputTag('softPFElectronsTagInfos'),
      ),
   mvaName = cms.string('BDT'),
   useCondDB = cms.bool(False),
   gbrForestLabel = cms.string(''),
   useGBRForest = cms.bool(True),
   useAdaBoost = cms.bool(False)
   )

charmTagsComputerCvsL.slComputerCfg.correctVertexMass = False

charmTagsComputerCvsB = charmTagsComputerCvsL.clone(
   weightFile = cms.FileInPath('RecoBTag/CTagging/data/c_vs_b_sklearn.weight.xml'),   
   variables = c_vs_b_vars_vpset
   )

#Negative Taggers
charmTagsNegativeComputerCvsL = charmTagsComputerCvsL.clone(
   vertexFlip = cms.bool(True),
   trackFlip = cms.bool(True)
)

charmTagsNegativeComputerCvsL.trackSelection.sip3dSigMax = 0
charmTagsNegativeComputerCvsL.trackPseudoSelection.sip3dSigMax = 0
charmTagsNegativeComputerCvsL.trackPseudoSelection.sip2dSigMin = -99999.9
charmTagsNegativeComputerCvsL.trackPseudoSelection.sip2dSigMax = -2.0

charmTagsNegativeComputerCvsB = charmTagsComputerCvsB.clone(
   vertexFlip = cms.bool(True),
   trackFlip = cms.bool(True)
)

charmTagsNegativeComputerCvsB.trackSelection.sip3dSigMax = 0
charmTagsNegativeComputerCvsB.trackPseudoSelection.sip3dSigMax = 0
charmTagsNegativeComputerCvsB.trackPseudoSelection.sip2dSigMin = -99999.9
charmTagsNegativeComputerCvsB.trackPseudoSelection.sip2dSigMax = -2.0

#Positive Taggers
charmTagsPositiveComputerCvsL = charmTagsComputerCvsL.clone(
)

charmTagsPositiveComputerCvsL.trackSelection.sip3dSigMin = 0
charmTagsPositiveComputerCvsL.trackPseudoSelection.sip3dSigMin = 0

charmTagsPositiveComputerCvsB = charmTagsComputerCvsB.clone(
)

charmTagsPositiveComputerCvsB.trackSelection.sip3dSigMin = 0
charmTagsPositiveComputerCvsB.trackPseudoSelection.sip3dSigMin = 0

