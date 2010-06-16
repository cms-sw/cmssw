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
    '/store/relval/CMSSW_3_8_0_pre2/RelValZEE/GEN-SIM-RECO/MC_38Y_V0-v1/0004/CC052912-0274-DF11-903E-002618943904.root',
    '/store/relval/CMSSW_3_8_0_pre2/RelValZEE/GEN-SIM-RECO/MC_38Y_V0-v1/0004/7E0EA211-2474-DF11-9EEC-001A92971B9A.root',
    '/store/relval/CMSSW_3_8_0_pre2/RelValZEE/GEN-SIM-RECO/MC_38Y_V0-v1/0004/58A6D390-FC73-DF11-B9A2-0018F3D096DC.root',
    '/store/relval/CMSSW_3_8_0_pre2/RelValZEE/GEN-SIM-RECO/MC_38Y_V0-v1/0004/462B4200-0174-DF11-B2C1-0018F3D096D4.root',
    '/store/relval/CMSSW_3_8_0_pre2/RelValZEE/GEN-SIM-RECO/MC_38Y_V0-v1/0004/069A2F8A-0074-DF11-9EED-0018F3D096AA.root'
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

#process.eIDClassesLoose = eidCutBasedClassesExt.clone()
#process.eIDClassesLoose.electronQuality = 'loose'

#process.eIDClassesMedium = eidCutBasedClassesExt.clone()
#process.eIDClassesMedium.electronQuality = 'medium'

#process.eIDClassesTight = eidCutBasedClassesExt.clone()
#process.eIDClassesTight.electronQuality = 'tight'

eIDSequence = cms.Sequence(process.eIDRobustLoose+
                           process.eIDRobustLooseV00+
                           process.eIDRobustTight+
                           process.eIDRobustHighEnergy+
                           process.eIDLoose+
                           process.eIDTight)
#                           process.eIDClassesLoose+ 
#                           process.eIDClassesMedium+ 
#                           process.eIDClassesTight )

process.p = cms.Path(eIDSequence)

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 
					   'keep *_gsfElectrons_*_*',
					   'keep *_eIDRobustLoose_*_*',
					   'keep *_eIDRobustLooseV00_*_*',
					   'keep *_eIDRobustTight_*_*',
					   'keep *_eIDRobustHighEnergy_*_*',
					   'keep *_eIDLoose_*_*',
					   'keep *_eIDTight_*_*'),
#					   'keep *_eIDClassesLoose_*_*',
#					   'keep *_eIDClassesMedium_*_*',
#					   'keep *_eIDClassesTight_*_*'),

    fileName = cms.untracked.string('electrons.root')
)

process.outpath = cms.EndPath(process.out)

