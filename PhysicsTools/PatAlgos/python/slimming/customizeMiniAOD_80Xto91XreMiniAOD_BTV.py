import FWCore.ParameterSet.Config as cms
from pdb  import set_trace

def customize4BTV(process, verbose=False):
   ' -- Add DeepCSV on miniAOD only -- '
   #set_trace()
   process.load('RecoBTag.Configuration.RecoBTag_cff')
   if not hasattr(process, 'patJets'):
      raise RuntimeError('The custimization from BTV requires patJets to be loaded in the process')
   
   #task black magic to force the modules to be run
   process.pfBTaggingTask.add(process.patJets)
   process.makePatJetsTask.replace(process.patJets, process.pfBTaggingTask)
   
   return process
