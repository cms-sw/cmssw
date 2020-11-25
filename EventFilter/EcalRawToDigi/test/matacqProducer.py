import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("EventFilter.EcalRawToDigi.ecalMatacq_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('WARNING')
    ),
    suppressInfo = cms.untracked.vstring('ecalMatacq')
)

#process.source = cms.Source("NewEventStreamFileReader",
#    debugFlag = cms.untracked.bool(True),
#    debugVebosity = cms.untracked.uint32(0),
#    fileNames = cms.untracked.vstring('/store/data/GlobalCRAFT1/Calibration/000/071/133/GlobalCRAFT1.00071133.0004.Calibration.storageManager.00.0000.dat')
#)
process.source = cms.Source ("PoolSource",
                     fileNames = cms.untracked.vstring('/store/data/Commissioning08/TestEnables/RAW/v1/000/069/071/688C2E1E-AEA9-DD11-A8EF-001D09F25041.root',
                                                       '/store/data/Commissioning08/TestEnables/RAW/v1/000/069/071/C0CFB816-AEA9-DD11-B59B-001D09F2426D.root'),
                     secondaryFileNames = cms.untracked.vstring(),
                     skipEvents = cms.untracked.uint32(800)
                     )                 

#process.source = cms.Source("PoolSource",
#                    fileNames = cms.untracked.vstring('input.root')
#                    )


#source =  cms.Source("LmfSource",
#                     fileNames = cms.untracked.vstring('input.lmf')
#                     )

process.o1 = cms.OutputModule("PoolOutputModule",
    compressionLevel = cms.untracked.int32(1),
    outputCommands = cms.untracked.vstring('keep *'),
    fileName = cms.untracked.string('output.root')
)

#histograming
process.hist = cms.EDAnalyzer("EcalMatacqHist2",
     firstTimePlotEvent = cms.untracked.int32(1),
     nTimePlots = cms.untracked.int32(10),
     matacqProducer = cms.string("ecalMatacq"),
     outputRootFile = cms.untracked.string('matacqHist.root')
)


process.p = cms.Path(process.ecalMatacq*process.hist)
process.outpath = cms.EndPath(process.o1)

#Matacq input data files:
process.ecalMatacq.fileNames = [
    'rfio:///castor/cern.ch/cms/store/data/Matacq2008Ecal/%run_subdir%/matacq.%run_number%.dat'
    ]

#performance timing
process.ecalMatacq.timing = True
process.ecalMatacq.disabled = False

#debugging verbosity
process.ecalMatacq.verbosity = 1


