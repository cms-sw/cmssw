import FWCore.ParameterSet.Config as cms

from RecoEcal.EgammaClusterProducers.geometryForClustering_cff import *
from RecoEgamma.ElectronIdentification.likelihoodPdfsDB_cfi import *
from RecoEgamma.ElectronIdentification.likelihoodESetup_cfi import *
from RecoEgamma.ElectronIdentification.neuralNetElectronId_cfi import *

from RecoEgamma.ElectronIdentification.electronIdCutBasedExt_cfi        import eidCutBasedExt
from RecoEgamma.ElectronIdentification.electronIdCutBasedClassesExt_cfi import eidCutBasedClassesExt as eidPtdrExt ## PTDR is how people know it
from RecoEgamma.ElectronIdentification.electronIdNeuralNetExt_cfi       import eidNeuralNetExt
from RecoEgamma.ElectronIdentification.electronIdLikelihoodExt_cfi      import eidLikelihoodExt 

# Temporary bugfix waiting for RecoEgamma/ElectronIdentification to be fixed upstream
electronIdPdfs.connect = 'sqlite_fip:CondCore/SQLiteData/data/electronIdLikelihoodTkIsolated.db'

eidPtdrExtPAT = eidPtdrExt.copy();  
eidPtdrExtPAT.src = cms.InputTag("allLayer0Electrons")
electronIdPTDRLoose  = eidPtdrExtPAT.copy(); electronIdPTDRLoose.electronQuality  = cms.string('loose')
electronIdPTDRMedium = eidPtdrExtPAT.copy(); electronIdPTDRMedium.electronQuality = cms.string('medium')
electronIdPTDRTight  = eidPtdrExtPAT.copy(); electronIdPTDRTight.electronQuality  = cms.string('tight')

eidCutBasedExtPAT = eidCutBasedExt.copy();  
eidCutBasedExtPAT.src = cms.InputTag("allLayer0Electrons")
electronIdCutBasedRobust = eidCutBasedExtPAT.copy(); electronIdCutBasedRobust.electronQuality = 'robust'
electronIdCutBasedLoose  = eidCutBasedExtPAT.copy(); electronIdCutBasedLoose.electronQuality  = 'loose'
electronIdCutBasedTight  = eidCutBasedExtPAT.copy(); electronIdCutBasedTight.electronQuality  = 'tight'

electronIdLikelihood = eidLikelihoodExt.copy(); 
electronIdLikelihood.src = cms.InputTag("allLayer0Electrons")

electronIdNeuralNet  = eidNeuralNetExt.copy(); 
electronIdNeuralNet.src = cms.InputTag("allLayer0Electrons")

patElectronId = cms.Sequence(electronIdPTDRLoose * 
                             electronIdPTDRMedium * 
                             electronIdPTDRTight * 
                             electronIdCutBasedRobust * 
                             electronIdCutBasedLoose * 
                             electronIdCutBasedTight * 
                             electronIdLikelihood * 
                             electronIdNeuralNet)
