from FWCore.ParameterSet.Config import *

process = cms.Process("runElectronID")

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load("Configuration.StandardSequences.MagneticField_cff")

from Geometry.CaloEventSetup.CaloTopology_cfi import *

process.maxEvents = cms.untracked.PSet(
   input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_10_0_pre2/RelValZEE/GEN-SIM-RECO/START39_V3-v1/0061/A0E0AF74-F5E2-DF11-A9BC-002618943939.root',
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

