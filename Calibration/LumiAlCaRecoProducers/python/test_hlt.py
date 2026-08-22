#########################
## standalone config to test PCC modules
## Jose Benitez, Attila Radl, Alexey Shevelev
#########################
import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run2_2018_cff import Run2_2018

process = cms.Process('PCC',Run2_2018)

process.source = cms.Source("PoolSource",
fileNames = cms.untracked.vstring('file:/eos/home-a/alshevel/CMSSW_14_0_9_patch1/src/Calibration/LumiAlCaRecoProducers/python/input_pcc.root')
#fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/b/benitezj/public/BRIL/PCC/Run3Dev/Run2018D-AlCaLumiPixels-RAW-323702-D3FCD0FC-6328-B24E-AD3D-C22C55B968DD.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2) 
)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.AlCaRecoStreams_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')


from EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi import siPixelDigis
process.siPixelDigisForLumi = siPixelDigis.cpu.clone(
   InputLabel = "hltFEDSelectorLumiPixels"
)

from RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizerPreSplitting_cfi import siPixelClustersPreSplitting
process.siPixelClustersForLumi = siPixelClustersPreSplitting.cpu.clone(
    src = "siPixelDigisForLumi"
)

################################
#RECO->ALCAPCC 
process.alcaPCCEventProducer = cms.EDProducer("AlcaPCCEventProducer",
    pixelClusterLabel = cms.InputTag("siPixelClustersForLumi"),
    trigstring = cms.untracked.string(""),
    savePerROCInfo = cms.bool(True)
)

################################
##ALCAPCC->ALCARECO
process.alcaPCCIntegrator = cms.EDProducer("AlcaPCCIntegrator",
    AlcaPCCIntegratorParameters = cms.PSet(
        inputPccLabel = cms.string("alcaPCCEventProducer"),
        trigstring = cms.untracked.string(""),
        ProdInst = cms.string("")
    ),
)

################################
##ALCARECO->csv
process.rawPCCProd = cms.EDProducer("RawPCCProducer",
    RawPCCProducerParameters = cms.PSet(
        inputPccLabel = cms.string("alcaPCCIntegrator"),
        ProdInst = cms.string(""),
        outputProductName = cms.untracked.string("lumiInfo"),
        ApplyCorrections=cms.untracked.bool(False),
        saveCSVFile=cms.untracked.bool(True),
        modVeto=cms.vint32(),
        OutputValue = cms.untracked.string("Average"),
    )
)

#################################
# OutPath products
process.ALCARECOStreamPromptCalibProdPCC = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOPromptCalibProdPCC')
    ),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('ALCAPROMPT'),
        filterName = cms.untracked.string('PromptCalibProdPCC')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    fileName = cms.untracked.string('PCC.root'),
    outputCommands = cms.untracked.vstring('drop *', 
                                           #'keep *_hltFEDSelectorLumiPixels_*_*',
                                           #'keep *_siPixelDigisForLumi_*_*',
                                           #'keep *_siPixelClustersForLumi_*_*',
                                           'keep *_alcaPCCEventProducer_*_*',
                                           'keep *_alcaPCCIntegrator_*_*',
                                           'keep *_rawPCCProd_*_*'
                                       )
)

####################
### sequences/paths
process.seqALCARECOPromptCalibProdPCC = cms.Sequence(process.siPixelDigisForLumi+process.siPixelClustersForLumi+process.alcaPCCEventProducer+process.alcaPCCIntegrator+process.rawPCCProd)
#process.seqALCARECOPromptCalibProdPCC = cms.Sequence(process.siPixelDigisForLumi+process.siPixelClustersForLumi+process.alcaPCCEventProducer+process.alcaPCCIntegrator)
#process.seqALCARECOPromptCalibProdPCC = cms.Sequence(process.siPixelDigisForLumi+process.siPixelClustersForLumi+process.alcaPCCEventProducer)

process.pathALCARECOPromptCalibProdPCC = cms.Path(process.seqALCARECOPromptCalibProdPCC)
process.ALCARECOStreamPromptCalibProdOutPath = cms.EndPath(process.ALCARECOStreamPromptCalibProdPCC)
process.schedule = cms.Schedule(*[ process.pathALCARECOPromptCalibProdPCC, process.ALCARECOStreamPromptCalibProdOutPath ])


#################################################
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        FwkJob = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        FwkReport = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000),
            reportEvery = cms.untracked.int32(100)
        ),
        FwkSummary = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000),
            reportEvery = cms.untracked.int32(100)
        ),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        Root_NoDictionary = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        noTimeStamps = cms.untracked.bool(False),
        threshold = cms.untracked.string('INFO'),
        enableStatistics = cms.untracked.bool(True),
        statisticsThreshold = cms.untracked.string('WARNING')
    ),
    debugModules = cms.untracked.vstring(),
    default = cms.untracked.PSet(),
    suppressDebug = cms.untracked.vstring(),
    suppressInfo = cms.untracked.vstring(),
    suppressWarning = cms.untracked.vstring()
)


process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
