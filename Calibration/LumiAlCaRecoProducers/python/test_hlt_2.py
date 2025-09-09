#########################
## standalone config to test PCC modules
## Jose Benitez, Attila Radl, Alexey Shevelev, modified by Peter Major
#########################
import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run2_2018_cff import Run2_2018

process = cms.Process('PCC',Run2_2018)

### Files run by Braden:
# 'file:/eos/cms/store/data/Run2024F/AlCaLumiPixelsCountsPrompt/ALCARECO/AlCaPCCZeroBias-PromptReco-v1/000/382/913/00000/242f6268-861c-4231-9cf8-b51c22aa7195.root'
# 'file:/eos/cms/store/data/Run2024F/AlCaLumiPixelsCountsPrompt/ALCARECO/AlCaPCCZeroBias-PromptReco-v1/000/382/913/00000/9491103d-34bf-441c-bd88-b7b3cd3d71a3.root'
# 'file:/eos/cms/store/data/Run2024F/AlCaLumiPixelsCountsPrompt/ALCARECO/AlCaPCCZeroBias-PromptReco-v1/000/382/913/00000/ac0dc5c4-c440-4c03-b876-e7731816355a.root'
# 'file:/eos/cms/store/data/Run2024F/AlCaLumiPixelsCountsPrompt/ALCARECO/AlCaPCCZeroBias-PromptReco-v1/000/382/913/00000/e644fd22-7a9f-4e3e-8edc-626aabea358d.root'
# 'file:/eos/cms/store/data/Run2024F/AlCaLumiPixelsCountsPrompt/ALCARECO/AlCaPCCZeroBias-PromptReco-v1/000/382/913/00000/edc8ddbc-de7e-4896-ad05-b63a1805d011.root'

process.source = cms.Source("PoolSource",
fileNames = cms.untracked.vstring('file:/eos/home-a/alshevel/CMSSW_14_0_9_patch1/src/Calibration/LumiAlCaRecoProducers/python/input_pcc.root')
#fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/b/benitezj/public/BRIL/PCC/Run3Dev/Run2018D-AlCaLumiPixels-RAW-323702-D3FCD0FC-6328-B24E-AD3D-C22C55B968DD.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000) 
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
    # savePerROCInfo = cms.bool(True)
    savePerROCInfo = cms.bool(False)
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

################################
##ALCARECO->csv
process.dynamicVetoProd = cms.EDProducer("DynamicVetoProducer",
    DynamicVetoProducerParameters = cms.PSet(
        inputPccLabel = cms.string("alcaPCCIntegrator"),
        prodInst = cms.string(""),
        outputProductName = cms.untracked.string("alcaPccVetoList"),
        BaseVeto=cms.vint32(),
        SaveBaseVeto=cms.bool(False),
        FractionalResponse_modID=cms.vint32(),
        FractionalResponse_value=cms.vdouble(),
        ModuleListRing1=cms.untracked.vint32(),
        MinimumLSCount=cms.untracked.int32(-1),
        StdMultiplyier1=cms.double(3.0),
        StdMultiplyier2=cms.double(3.0),
        FractionThreshold2=cms.double(0.02),
        StdMultiplyier3=cms.double(3.0),
        FilterLevel=cms.int32(3),
        SavePlots=cms.untracked.bool(True),
        CoutOn=cms.untracked.bool(True),
        SaveCSVFile=cms.untracked.bool(True),
        CsvFileName=cms.untracked.string("dynamicVetoProducer_Run2test.csv"),
    )
)



with open("../data/minimal_veto-2024.txt") as f: 
    process.dynamicVetoProd.DynamicVetoProducerParameters.BaseVeto.extend([ int(v) for v in f.readlines()])

with open("../data/minimal_veto_frac_response-2024.txt") as f: 
    tmp = [ v.split(",") for v in f.readlines()]
    moduleID          = [ int(l[0]) for l in tmp[1:]]
    ractionalResponse = [ float(l[1]) for l in tmp[1:]]
    process.dynamicVetoProd.DynamicVetoProducerParameters.FractionalResponse_modID.extend(moduleID)
    process.dynamicVetoProd.DynamicVetoProducerParameters.FractionalResponse_value.extend(ractionalResponse)



process.dynamicVetoProd.DynamicVetoProducerParameters.ModuleListRing1.extend([
  344282116, 344283140, 344286212, 344941572, 352588804, 352589828, 352592900, 353215492, 344724484, 344725508, 344728580, 344729604, 344732676, 344733700, 344736772, 344737796, 
  352596996, 352598020, 352601092, 353227780, 352605188, 353235972, 352609284, 352610308, 344740868, 344741892, 344744964, 344745988, 344749060, 344750084, 344753156, 344754180, 
  352613380, 353244164, 352617476, 352618500, 352621572, 353256452, 352625668, 353268740, 344757252, 344758276, 344761348, 344762372, 344765444, 344766468, 344769540, 344770564, 
  352629764, 353276932, 352633860, 352634884, 352637956, 353285124, 352642052, 352643076, 344773636, 344774660, 344777732, 344778756, 344781828, 344782852, 344785924, 344786948, 
  352646148, 353297412, 352650244, 353305604, 352654340, 352655364, 352658436, 353313796, 344790020, 344791044, 344794116, 344795140, 344798212, 344799236, 344802308, 344803332, 
  352662532, 352663556, 352666628, 353326084, 352670724, 353338372, 352674820, 353207300, 344806404, 344807428, 344810500, 344811524, 344462340, 344463364, 344466436, 344467460, 
  352850948, 352851972, 352855044, 352856068, 352859140, 352860164, 352863236, 352864260, 344470532, 344471556, 344474628, 344475652, 344478724, 344479748, 344482820, 344483844, 
  352867332, 352868356, 352871428, 352872452, 352875524, 352876548, 352879620, 352880644, 344486916, 344487940, 344491012, 344492036, 344495108, 344496132, 344499204, 344500228, 
  352883716, 352884740, 352887812, 352888836, 352891908, 352892932, 352896004, 352897028, 344503300, 344504324, 344507396, 344508420, 344511492, 344512516, 344515588, 344516612, 
  352900100, 352901124, 352904196, 352905220, 352908292, 352909316, 352912388, 352913412, 344519684, 344520708, 344523780, 344524804, 344527876, 344528900, 344531972, 344532996, 
  352916484, 352917508, 352920580, 352921604, 352924676, 352925700, 352928772, 352929796, 344536068, 344537092, 344540164, 344541188, 344544260, 344545284, 344548356, 344549380, 
  352932868, 352933892, 352936964, 352937988, 353113092, 353114116, 353117188, 353118212, 344200196, 344201220, 344204292, 344949764, 344208388, 344818692, 344212484, 344830980, 
  353121284, 353122308, 353125380, 353126404, 353129476, 353130500, 353133572, 353134596, 344216580, 344217604, 344220676, 344843268, 344224772, 344225796, 344228868, 344851460, 
  353137668, 353138692, 353141764, 353142788, 353145860, 353146884, 353149956, 353150980, 344232964, 344859652, 344237060, 344238084, 344241156, 344871940, 344245252, 344246276, 
  353154052, 353155076, 353158148, 353159172, 353162244, 353163268, 353166340, 353167364, 344249348, 344880132, 344253444, 344888324, 344257540, 344900612, 344261636, 344262660, 
  353170436, 353171460, 353174532, 353175556, 353178628, 353179652, 353182724, 353183748, 344265732, 344912900, 344269828, 344270852, 344273924, 344921092, 344278020, 344929284, 
  353186820, 353187844, 353190916, 353191940, 353195012, 353196036, 353199108, 353200132 
])

#####################################

# process.dynamicVetoProd_old = cms.EDProducer("DynamicVetoProducer_old",
#     DynamicVetoProducer_oldParameters = cms.PSet(
#         inputPccLabel = cms.string("alcaPCCIntegrator"),
#         BaseVeto=cms.vint32(),
#         MinimumLSCount=cms.untracked.int32(50),
#         StdMultiplyier1=cms.double(3.0),
#         StdMultiplyier2=cms.double(3.0),
#         SaveCSVFile=cms.untracked.bool(True),
#         CsvFileName=cms.untracked.string("dynamicVetoProducer.csv"),
#     )
# )

# process.load("CondCore.CondDB.CondDB_cfi")
# process.CondDB.connect = "sqlite_file:PCC_Veto.db" # Output SQLite file
# process.PoolDBOutputService = cms.Service(
#     "PoolDBOutputService", process.CondDB,
#     toPut = cms.VPSet(
#         cms.PSet(
#             record = cms.string('PccVetoListRcd'),
#             tag = cms.string('TestVeto')
#         )
#     ),
#     loadBlobStreamer = cms.untracked.bool(False),
#     timetype   = cms.untracked.string('runnumber'),
#     DBParameters=cms.PSet(messageLevel=cms.untracked.int32(0))
# )

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
    fileName = cms.untracked.string('PCC_Run2test.root'),
    outputCommands = cms.untracked.vstring('drop *', 
                                           #'keep *_hltFEDSelectorLumiPixels_*_*',
                                           #'keep *_siPixelDigisForLumi_*_*',
                                           #'keep *_siPixelClustersForLumi_*_*',
                                           'keep *_alcaPCCEventProducer_*_*',
                                           'keep *_alcaPCCIntegrator_*_*',
                                           'keep *_rawPCCProd_*_*',
                                        #    'keep *_dynamicVetoProd_*_*',
                                        #    'keep *_dynamicVetoProd_old_*_*',
                                       )
)

####################
### sequences/paths
process.seqALCARECOPromptCalibProdPCC = cms.Sequence(
    process.siPixelDigisForLumi
    +process.siPixelClustersForLumi
    +process.alcaPCCEventProducer
    +process.alcaPCCIntegrator
    +process.rawPCCProd
    +process.dynamicVetoProd
    # +process.dynamicVetoProd_old
    )
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
