import FWCore.ParameterSet.Config as cms
import RecoBTag.SecondaryVertex.candidateCombinedSecondaryVertexV2Computer_cfi as sl_cfg 
from RecoBTag.Combined.hiRun2LegacyCSVv2_trainingSettings import hiRun2LegacyCSVv2_vpset
from RecoBTag.Combined.hiRun2LegacyCSVv2_helpers import get_vars


hiRun2LegacyCSVv2Computer = cms.ESProducer(
   'hiRun2LegacyCSVv2ESProducer',
   slComputerCfg = cms.PSet(
      **sl_cfg.candidateCombinedSecondaryVertexV2Computer.parameters_()
      ),
   weightFile = cms.FileInPath('RecoBTag/Combined/data/TMVA_Btag_CsJets_PbPb2018_BDTG.weights.xml'),

   variables = hiRun2LegacyCSVv2_vpset,
   computer = cms.ESInputTag('dummy:dummy'),
   tagInfos = cms.VInputTag(
      cms.InputTag('impactParameterTagInfos'),
      cms.InputTag('secondaryVertexFinderTagInfos'),
      ),
   mvaName = cms.string('BDT'),
   useCondDB = cms.bool(False),
   gbrForestLabel = cms.string(''),
   useGBRForest = cms.bool(True),
   useAdaBoost = cms.bool(False)
   )


