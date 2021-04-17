import FWCore.ParameterSet.Config as cms
import RecoBTag.SecondaryVertex.candidateCombinedSecondaryVertexV2Computer_cfi as sv_cfg 
from RecoBTag.Combined.heavyIonCSV_trainingSettings import heavyIonCSV_vpset

heavyIonCSVComputer = cms.ESProducer(
   'HeavyIonCSVESProducer',
   sv_cfg = cms.PSet(
      **sv_cfg.candidateCombinedSecondaryVertexV2Computer.parameters_()
      ),
   weightFile = cms.FileInPath('RecoBTag/Combined/data/TMVA_Btag_CsJets_PbPb2018_BDTG.weights.xml'),

   variables = heavyIonCSV_vpset,
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


