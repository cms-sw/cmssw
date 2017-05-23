import FWCore.ParameterSet.Config as cms
from pdb  import set_trace

def customize4BTV(process, verbose=False):
   ' -- Add DeepCSV on miniAOD only -- '
   #set_trace()
   if not hasattr(process, 'patJets'):
      raise RuntimeError('The custimization from BTV requires patJets to be loaded in the process')
   
   process.load('RecoBTag.Combined.deepFlavour_cff')
   #task black magic to force the modules to be run
   process.pfDeepFlavourTask.add(process.patJets)
   if not hasattr(process, 'pfImpactParameterTagInfos'):
      process.load('RecoBTag.ImpactParameter.pfImpactParameterTagInfos_cfi')
      process.pfDeepFlavourTask.add(process.pfImpactParameterTagInfos)
   if not hasattr(process, 'pfSecondaryVertexTagInfos'):
      process.load('RecoBTag.SecondaryVertex.pfSecondaryVertexTagInfos_cfi')
      process.pfDeepFlavourTask.add(process.pfSecondaryVertexTagInfos)
   if not hasattr( process, 'pfInclusiveSecondaryVertexFinderTagInfos'):
      process.load('RecoBTag.SecondaryVertex.pfInclusiveSecondaryVertexFinderTagInfos_cfi')
      process.pfDeepFlavourTask.add(process.pfInclusiveSecondaryVertexFinderTagInfos)
   process.makePatJetsTask.replace(process.patJets, process.pfDeepFlavourTask)
   
   return process
