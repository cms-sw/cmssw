import FWCore.ParameterSet.Config as cms
process = cms.Process("HLTBTAG")
process.load("L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff")
process.load("DQMServices.Components.EDMtoMEConverter_cff")

process.load("HLTriggerOffline.Btag.HltBtagValidation_cff")
#process.load("HLTriggerOffline.Btag.HltBtagValidationFastSim_cff")
process.load("HLTriggerOffline.Btag.HltBtagPostValidation_cff")

#process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(500)   )

process.DQM_BTag = cms.Path(    process.hltbtagValidationSequence + process.HltBTagPostVal + process.dqmSaver)

import sys
import Utilities.General.cmssw_das_client as cmssw_das_client
def add_rawRelVals(process, inputName):   
   query='dataset='+inputName 
   dataset = cmssw_das_client.get_data(query, limit = 0)
   if not dataset:
      raise RuntimeError(
         'Das returned no dataset parent of the input file: %s \n'
         'The parenthood is needed to add RAW secondary input files' % process.source.fileNames[0]
         )
   for i in dataset['data']:
	try: n_files = i['dataset'][0]['num_file']
	except: pass
   raw_files = cmssw_das_client.get_data('file '+query, limit = 0)
   files = []
   for i in raw_files['data']:
	files.append( i['file'][0]['name'])
   
   raw_files = ['root://cms-xrd-global.cern.ch/'+str(i) for i in files]
   process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring(raw_files))
   return process

add_rawRelVals(process, str(sys.argv[-1]))

#process.source = cms.Source("PoolSource",
#	fileNames = cms.untracked.vstring(
#'root://cms-xrd-global.cern.ch//store/user/mdefranc/RelValTTbar_13/ttbarRelVal_noPU_3/180715_094624/0000/step2_50.root',
#)
#)



#Settings equivalent to 'RelVal' convention:
process.dqmSaver.convention = 'Offline'
process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)
process.dqmSaver.forceRunNumber = cms.untracked.int32(1)
process.dqmSaver.workflow = "/test/RelVal/TrigVal"
process.DQMStore.collateHistograms = False
process.DQMStore.verbose=0
process.options = cms.untracked.PSet(
	wantSummary	= cms.untracked.bool( True ),
	fileMode	= cms.untracked.string('FULLMERGE'),
	SkipEvent	= cms.untracked.vstring('ProductNotFound')
)

