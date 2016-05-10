import FWCore.ParameterSet.Config as cms
#use import as to mask it to process.load() 
import RecoBTag.SecondaryVertex.candidateCombinedSecondaryVertexSoftLeptonComputer_cfi as sl_cfg 
from RecoBTag.CTagging.training_settings import c_vs_l_vars_vpset, c_vs_b_vars_vpset
#from RecoBTag.SecondaryVertex.combinedSecondaryVertexCommon_cff import *

#
# Normal tagger
#

charmTagsComputerCvsL = cms.ESProducer(
   'CharmTaggerESProducer',
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

#
# Negative tagger
#

charmTagsNegativeComputerCvsL = charmTagsComputerCvsL.clone(
   vertexFlip = cms.bool(True),
   trackFlip = cms.bool(True)
)

charmTagsNegativeComputerCvsL.slComputerCfg.trackSelection.sip3dSigMax = 0
charmTagsNegativeComputerCvsL.slComputerCfg.trackPseudoSelection.sip3dSigMax = 0
charmTagsNegativeComputerCvsL.slComputerCfg.trackPseudoSelection.sip2dSigMin = -99999.9
charmTagsNegativeComputerCvsL.slComputerCfg.trackPseudoSelection.sip2dSigMax = -2.0

charmTagsNegativeComputerCvsB = charmTagsComputerCvsB.clone(
   vertexFlip = cms.bool(True),
   trackFlip = cms.bool(True)
)

charmTagsNegativeComputerCvsB.slComputerCfg.trackSelection.sip3dSigMax = 0
charmTagsNegativeComputerCvsB.slComputerCfg.trackPseudoSelection.sip3dSigMax = 0
charmTagsNegativeComputerCvsB.slComputerCfg.trackPseudoSelection.sip2dSigMin = -99999.9
charmTagsNegativeComputerCvsB.slComputerCfg.trackPseudoSelection.sip2dSigMax = -2.0

#
# Positive tagger
#

charmTagsPositiveComputerCvsL = charmTagsComputerCvsL.clone(
)

charmTagsPositiveComputerCvsL.slComputerCfg.trackSelection.sip3dSigMin = 0
charmTagsPositiveComputerCvsL.slComputerCfg.trackPseudoSelection.sip3dSigMin = 0

charmTagsPositiveComputerCvsB = charmTagsComputerCvsB.clone(
)

charmTagsPositiveComputerCvsB.slComputerCfg.trackSelection.sip3dSigMin = 0
charmTagsPositiveComputerCvsB.slComputerCfg.trackPseudoSelection.sip3dSigMin = 0

