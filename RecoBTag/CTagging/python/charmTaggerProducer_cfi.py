import FWCore.ParameterSet.Config as cms
from RecoBTag.CTagging.trainingvars import get_var_pset
#use import as to mask it to process.load() (hopefully)
import RecoBTag.SecondaryVertex.candidateCombinedSecondaryVertexSoftLeptonComputer_cfi as sl_cfg 
from RecoBTag.CTagging.helpers import get_vars

charmTagsComputerCvsL = cms.ESProducer(
   'CharmTaggerESProducer',
   #clone the cfg only
   slComputerCfg = cms.PSet(
      **sl_cfg.candidateCombinedSecondaryVertexSoftLeptonComputer.parameters_()
      ),
   weightFile = cms.FileInPath('RecoBTag/CTagging/data/c_vs_udsg.weight.xml'),
   variables = cms.VPSet(
      #would be cool if we could directly read the TMVA wieight file and get the information from there
      #the main problem is unpack cms.FileInPath from python, to make it grid-safe
      ),
   computer = cms.ESInputTag('combinedSecondaryVertexSoftLeptonComputer'),
   tagInfos = cms.VInputTag(
      cms.InputTag('pfImpactParameterTagInfos'),
      cms.InputTag('pfInclusiveSecondaryVertexFinderCtagLTagInfos'),
      cms.InputTag('softPFMuonsTagInfos'),
      cms.InputTag('softPFElectronsTagInfos'),
      )
   )

charmTagsComputerCvsL.variables = cms.VPSet( 
   *get_vars(
      charmTagsComputerCvsL.weightFile.value()
      )
    )

charmTagsComputerCvsB = charmTagsComputerCvsL.clone(
   weightFile = cms.FileInPath('RecoBTag/CTagging/data/c_vs_b.weight.xml'),   
   variables = cms.VPSet()
   )

charmTagsComputerCvsB.variables = cms.VPSet( 
   *get_vars(
      charmTagsComputerCvsB.weightFile.value()
      )
    )
