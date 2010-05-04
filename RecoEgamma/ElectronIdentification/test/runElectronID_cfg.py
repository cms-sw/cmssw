from FWCore.ParameterSet.Config import *

process = cms.Process("runElectronID")

process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/Services_cff')
process.load('Configuration/StandardSequences/Geometry_cff')
process.load("Configuration.StandardSequences.MagneticField_cff")

from Geometry.CaloEventSetup.CaloTopology_cfi import *

process.maxEvents = cms.untracked.PSet(
   input = cms.untracked.int32(200)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_7_0_pre2/RelValZEE/GEN-SIM-RECO/START37_V1-v1/0018/9C59B317-F752-DF11-94D0-0026189438A7.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValZEE/GEN-SIM-RECO/START37_V1-v1/0017/DC6EC4F6-9D52-DF11-AF94-00261894383E.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValZEE/GEN-SIM-RECO/START37_V1-v1/0017/D203FD36-9F52-DF11-9631-00261894383F.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValZEE/GEN-SIM-RECO/START37_V1-v1/0017/B89969B0-9C52-DF11-9E8B-002618943946.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValZEE/GEN-SIM-RECO/START37_V1-v1/0017/9CDDFFA7-9B52-DF11-8F03-00261894389C.root'
    ),
    secondaryFileNames = cms.untracked.vstring (
   )
)

process.load("RecoEgamma.ElectronIdentification.electronIdCutBasedExt_cfi")
from RecoEgamma.ElectronIdentification.electronIdCutBasedExt_cfi import *
process.load("RecoEgamma.ElectronIdentification.electronIdCutBasedClassesExt_cfi")
from RecoEgamma.ElectronIdentification.electronIdCutBasedClassesExt_cfi import *

process.eIDRobustLoose = eidCutBasedExt.clone()
process.eIDRobustLoose.electronIDType = 'robust'
process.eIDRobustLoose.electronQuality = 'loose'

process.eIDRobustLooseV00 = eidCutBasedExt.clone()
process.eIDRobustLooseV00.electronIDType = 'robust'
process.eIDRobustLooseV00.electronQuality = 'loose'
process.eIDRobustLooseV00.electronVersion = 'V00'

process.eIDRobustTight = eidCutBasedExt.clone()
process.eIDRobustTight.electronIDType = 'robust'
process.eIDRobustTight.electronQuality = 'tight'

process.eIDRobustHighEnergy = eidCutBasedExt.clone()
process.eIDRobustHighEnergy.electronIDType = 'robust'
process.eIDRobustHighEnergy.electronQuality = 'highenergy'

process.eIDLoose = eidCutBasedExt.clone()
process.eIDLoose.electronIDType = 'classbased'
process.eIDLoose.electronQuality = 'loose'

process.eIDTight = eidCutBasedExt.clone()
process.eIDTight.electronIDType = 'classbased'
process.eIDTight.electronQuality = 'tight'

process.eIDClassesLoose = eidCutBasedClassesExt.clone()
process.eIDClassesLoose.electronQuality = 'loose'

process.eIDClassesMedium = eidCutBasedClassesExt.clone()
process.eIDClassesMedium.electronQuality = 'medium'

process.eIDClassesTight = eidCutBasedClassesExt.clone()
process.eIDClassesTight.electronQuality = 'tight'

eIDSequence = cms.Sequence(process.eIDRobustLoose+
                           process.eIDRobustLooseV00+
                           process.eIDRobustTight+
                           process.eIDRobustHighEnergy+
                           process.eIDLoose+
                           process.eIDTight+
                           process.eIDClassesLoose+ 
                           process.eIDClassesMedium+ 
                           process.eIDClassesTight )

process.p = cms.Path(eIDSequence)

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 
					   'keep *_gsfElectrons_*_*',
					   'keep *_eIDRobustLoose_*_*',
					   'keep *_eIDRobustLooseV00_*_*',
					   'keep *_eIDRobustTight_*_*',
					   'keep *_eIDRobustHighEnergy_*_*',
					   'keep *_eIDLoose_*_*',
					   'keep *_eIDTight_*_*',
					   'keep *_eIDClassesLoose_*_*',
					   'keep *_eIDClassesMedium_*_*',
					   'keep *_eIDClassesTight_*_*'),

    fileName = cms.untracked.string('electrons.root')
)

process.outpath = cms.EndPath(process.out)

