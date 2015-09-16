import FWCore.ParameterSet.Config as cms
#use import as to mask it to process.load() 
import RecoBTag.SecondaryVertex.candidateCombinedSecondaryVertexSoftLeptonComputer_cfi as sl_cfg 
from RecoBTag.CTagging.training_settings import c_vs_l_vars_vpset, c_vs_b_vars_vpset

charmTagsComputerCvsL = cms.ESProducer(
   'CharmTaggerESProducer',
   #clone the cfg only
   slComputerCfg = cms.PSet(
      **sl_cfg.candidateCombinedSecondaryVertexSoftLeptonComputer.parameters_()
      ),
   weightFile = cms.FileInPath('RecoBTag/CTagging/data/c_vs_udsg.weight.xml'),
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

charmTagsComputerCvsB = charmTagsComputerCvsL.clone(
   weightFile = cms.FileInPath('RecoBTag/CTagging/data/c_vs_b.weight.xml'),   
   variables = c_vs_b_vars_vpset
   )

