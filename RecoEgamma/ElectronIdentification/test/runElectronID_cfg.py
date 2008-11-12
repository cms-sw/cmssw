from FWCore.ParameterSet.Config import *

process = cms.Process("runElectronID")

process.extend(include("FWCore/MessageLogger/data/MessageLogger.cfi"))
process.extend(include("RecoEcal/EgammaClusterProducers/data/geometryForClustering.cff"))

process.maxEvents = cms.untracked.PSet(
#    input = cms.untracked.int32(-1)
   input = cms.untracked.int32(100)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_2_2_0_pre1/RelValSingleElectronPt35/GEN-SIM-RECO/IDEAL_V9_v1/0000/5A5904B4-19AF-DD11-91FE-001617DBD332.root',
    '/store/relval/CMSSW_2_2_0_pre1/RelValSingleElectronPt35/GEN-SIM-RECO/IDEAL_V9_v1/0000/7CE8895D-A8AE-DD11-9A40-000423D98AF0.root'
   ),
    secondaryFileNames = cms.untracked.vstring (
   )
)

process.load("RecoEgamma.ElectronIdentification.electronIdCutBasedExt_cfi")
from RecoEgamma.ElectronIdentification.electronIdCutBasedExt_cfi import *

process.eIDRobustLoose = eidCutBasedExt.clone()
process.eIDRobustLoose.electronQuality = 'robust'

process.eIDRobustTight = eidCutBasedExt.clone()
process.eIDRobustTight.electronQuality = 'robust'
process.eIDRobustTight.robustEleIDCuts.barrel = [0.015, 0.0092, 0.020, 0.0025]
process.eIDRobustTight.robustEleIDCuts.endcap = [0.018, 0.025, 0.020, 0.0040]

process.eIDRobustHighEnergy = eidCutBasedExt.clone()
process.eIDRobustHighEnergy.electronQuality = 'robust'
process.eIDRobustHighEnergy.robustEleIDCuts.barrel = [0.050, 0.011, 0.090, 0.005]
process.eIDRobustHighEnergy.robustEleIDCuts.endcap = [0.100, 0.0275, 0.090, 0.007]

process.eIDLoose = eidCutBasedExt.clone()
process.eIDLoose.electronQuality = 'loose'

process.eIDTight = eidCutBasedExt.clone()
process.eIDTight.electronQuality = 'tight'

eIDSequence = cms.Sequence(process.eIDRobustLoose+process.eIDRobustTight+process.eIDRobustHighEnergy+process.eIDLoose+process.eIDTight)
process.p = cms.Path(eIDSequence)

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 
					   'keep *_pixelMatchGsfElectrons_*_*',
					   'keep *_eIDRobustLoose_*_*',
					   'keep *_eIDRobustTight_*_*',
					   'keep *_eIDRobustHighEnergy_*_*',
					   'keep *_eIDLoose_*_*',
					   'keep *_eIDTight_*_*'),

    fileName = cms.untracked.string('electrons.root')
)

process.outpath = cms.EndPath(process.out)

