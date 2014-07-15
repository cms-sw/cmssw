import FWCore.ParameterSet.Config as cms

""" This python cfg converts a plain DQM root file in a EDM file that can be used to run the SiStrip bad channel calibration as done @ Tier0"""



process = cms.Process("CONV")

#process.load("DQMServices.Core.test.MessageLogger_cfi")
process.load('Configuration.EventContent.EventContent_cff')
process.load("DQMServices.Core.DQM_cfg")
#process.DQMStore.collateHistograms = cms.untracked.bool(True)



process.fileReader = cms.EDAnalyzer("DQMRootFileReader",
                                    RootFileName = cms.untracked.string
                                    ('/build/cerminar/debugpcl/CMSSW_5_3_16/src/Calibration/TkAlCaRecoProducers/test/DQM_V0001_R000160819__StreamExpress__Run2011A-Express-v1__DQM.root')
                                    )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )


runNumber = int(process.fileReader.RootFileName.value().split('/')[-1].split('_')[2].lstrip("R"))
print "Run number extracted from file name:",runNumber

process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(runNumber),
                            )




process.ALCARECOStreamSiStripPCLHistos = cms.OutputModule("PoolOutputModule",
                                                          outputCommands = cms.untracked.vstring('drop *', 
                                                                                                 'keep *_MEtoEDMConvertSiStrip_*_*'),
                                                          fileName = cms.untracked.string('SiStripPCLHistos.root'),
                                                          dataset = cms.untracked.PSet(
                                                              filterName = cms.untracked.string(''),
                                                              dataTier = cms.untracked.string('ALCARECO')
                                                              ),
                                                          eventAutoFlushCompressedSize = cms.untracked.int32(5242880)
                                                          )

process.MEtoEDMConvertSiStrip = cms.EDProducer("MEtoEDMConverter",
                                               deleteAfterCopy = cms.untracked.bool(False),
                                               Verbosity = cms.untracked.int32(0),
                                               Frequency = cms.untracked.int32(50),
                                               Name = cms.untracked.string('MEtoEDMConverter'),
                                               MEPathToSave = cms.untracked.string('AlCaReco/SiStrip')
                                               )



process.path = cms.Path(process.fileReader)
process.ALCARECOStreamSiStripPCLHistosOutPath = cms.EndPath(process.ALCARECOStreamSiStripPCLHistos)
process.endjob_step = cms.EndPath(process.MEtoEDMConvertSiStrip)


process.schedule = cms.Schedule(process.path, process.endjob_step, process.ALCARECOStreamSiStripPCLHistosOutPath)
