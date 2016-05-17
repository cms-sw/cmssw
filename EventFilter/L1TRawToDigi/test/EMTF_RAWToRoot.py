## Initial script to convert from .dat files to root output with EDMCollections - AWB 29.01.16

import FWCore.ParameterSet.Config as cms

process = cms.Process("EMTF")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10000))
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(100)
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))

infiles = [

    ## eos ls store/express/Commissioning2016/ExpressCosmics/FEVT/Express-v1/000/264/374/00000/ - MWGR #1, 10.02.16
    ## eos ls store/express/Commissioning2016/ExpressCosmics/FEVT/Express-v1/000/264/479/00000/ - MWGR #1, 11.02.16
    ## eos ls store/express/Commissioning2016/ExpressCosmics/FEVT/Express-v1/000/264/593/00000/ - MWGR #1, 12.02.16
    ## eos ls store/express/Commissioning2016/ExpressCosmics/FEVT/Express-v1/000/266/537/00000/ - MWGR #3, 11.03.16, EMTF internal track BX off by +1
    
    ## *** MWGR #3 runs ***
    ## 266150, 266275, 266423 - Bad EMTF timing
    ## "March 10th, around 9pm, shifted comma gap delay ... leads to our DAQ data being shifted in the readout window by +1 BX.
    ## 266535, 266536, 266537, 266538 - Early morning March 11, no OMTF
    ## 266665, 266667, 266681 - All day March 12, overnight March 13

    ## Run 266537: out of 10k events, 338 tracks, 178 in BX +1 (phi 195 - 315, SP_TS = 3/4, 11/12)
    ## Run 266681: out of 10k events, 277 tracks, 148 in BX +1 (phi 195 - 330, SP_TS = 3/4, 11/12)

    ## DAS: dataset=/*/Commissioning2016*/RAW
    ## DAS: dataset=/*/Commissioning2016*/FEVT

    # 'root://eoscms//store/express/Commissioning2016/ExpressCosmics/FEVT/Express-v1/000/266/537/00000/A4367789-46E7-E511-9376-02163E014176.root',
    # 'root://eoscms//store/express/Commissioning2016/ExpressCosmics/FEVT/Express-v1/000/266/681/00000/A674A9F6-7DE9-E511-81C7-02163E0144A2.root',

    'root://eoscms//store/express/Commissioning2016/ExpressCosmics/FEVT/Express-v1/000/268/733/00000/5669AE2E-CAFC-E511-9402-02163E01452B.root',
    'root://eoscms//store/express/Commissioning2016/ExpressCosmics/FEVT/Express-v1/000/268/733/00000/A459B232-CAFC-E511-B119-02163E011856.root',

    # 'root://eoscms//store/express/Commissioning2016/ExpressCosmics/FEVT/Express-v1/000/266/537/00000/A4367789-46E7-E511-9376-02163E014176.root',
    # 'root://eoscms//store/express/Commissioning2016/ExpressCosmics/FEVT/Express-v1/000/266/537/00000/A840D46E-55E7-E511-A549-02163E011939.root',
    # 'root://eoscms//store/express/Commissioning2016/ExpressCosmics/FEVT/Express-v1/000/266/537/00000/A89B6F73-55E7-E511-9C18-02163E01381F.root',
    # 'root://eoscms//store/express/Commissioning2016/ExpressCosmics/FEVT/Express-v1/000/266/537/00000/B0CDFC69-4CE7-E511-9BB1-02163E011F3E.root',
    # 'root://eoscms//store/express/Commissioning2016/ExpressCosmics/FEVT/Express-v1/000/266/537/00000/B2FB6B80-4BE7-E511-9AA6-02163E012236.root',
    # 'root://eoscms//store/express/Commissioning2016/ExpressCosmics/FEVT/Express-v1/000/266/537/00000/B8AE537C-54E7-E511-9E62-02163E01396A.root',
    # 'root://eoscms//store/express/Commissioning2016/ExpressCosmics/FEVT/Express-v1/000/266/537/00000/B8C6436E-4BE7-E511-AD9F-02163E011FCE.root',
    # 'root://eoscms//store/express/Commissioning2016/ExpressCosmics/FEVT/Express-v1/000/266/537/00000/BA765DA3-45E7-E511-A29E-02163E01381F.root',
    # 'root://eoscms//store/express/Commissioning2016/ExpressCosmics/FEVT/Express-v1/000/266/537/00000/BAC60444-49E7-E511-A2CB-02163E01381F.root',
    # 'root://eoscms//store/express/Commissioning2016/ExpressCosmics/FEVT/Express-v1/000/266/537/00000/BCA322C2-50E7-E511-89EC-02163E011939.root',
    # 'root://eoscms//store/express/Commissioning2016/ExpressCosmics/FEVT/Express-v1/000/266/537/00000/C477D6A6-4AE7-E511-AFAE-02163E011FCE.root',
    # 'root://eoscms//store/express/Commissioning2016/ExpressCosmics/FEVT/Express-v1/000/266/537/00000/C682A47E-47E7-E511-AF01-02163E0134CD.root',
    # 'root://eoscms//store/express/Commissioning2016/ExpressCosmics/FEVT/Express-v1/000/266/537/00000/C85F11A8-4AE7-E511-B380-02163E01381F.root',
    # 'root://eoscms//store/express/Commissioning2016/ExpressCosmics/FEVT/Express-v1/000/266/537/00000/CC92A82C-4DE7-E511-97D7-02163E013613.root',

    # 'root://eoscms//store/express/Commissioning2016/ExpressCosmics/FEVT/Express-v1/000/264/593/00000/002BAC1F-EBD1-E511-A705-02163E0144B7.root',
    # 'root://eoscms//store/express/Commissioning2016/ExpressCosmics/FEVT/Express-v1/000/264/593/00000/004B8353-10D2-E511-B539-02163E0136DC.root',
    # 'root://eoscms//store/express/Commissioning2016/ExpressCosmics/FEVT/Express-v1/000/264/593/00000/00545D78-30D2-E511-8295-02163E013703.root',
    # 'root://eoscms//store/express/Commissioning2016/ExpressCosmics/FEVT/Express-v1/000/264/593/00000/00B9326E-0AD2-E511-8C9A-02163E0136F8.root',
]

fNames = cms.untracked.vstring('file:/afs/cern.ch/work/a/abrinke1/public/EMTF/miniDAQ/dat_dumps/2015_12_13/263758/run263758_ls0025_streamA_StorageManager.dat')


process.source = cms.Source(
    "PoolSource",
    # fileNames = fNames,
    fileNames = cms.untracked.vstring(infiles)
    )


# process.source = cms.Source(
#     "NewEventStreamFileReader",
#     # fileNames = fNames,
#     fileNames = cms.untracked.vstring(infiles),
#     skipEvents=cms.untracked.uint32(123)
# )

# PostLS1 geometry used
process.load('Configuration.Geometry.GeometryExtended2015Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2015_cff')
############################
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

# ## Debug / error / warning message output
# process.MessageLogger = cms.Service(
#     "MessageLogger",
#     threshold  = cms.untracked.string('DEBUG'),
#     categories = cms.untracked.vstring('L1T'),
#     ## categories = cms.untracked.vstring('EMTF'),
#     debugModules = cms.untracked.vstring('*'),
#     )


# Dump raw data of payload as text
process.dump = cms.EDAnalyzer( 
    "DumpFEDRawDataProduct",
    label = cms.untracked.string("rawDataCollector"),
    # feds = cms.untracked.vint32(1402,813),
    # feds = cms.untracked.vint32(1402),
    ## Dump payload as text
    dumpPayload = cms.untracked.bool ( True )
)

process.unpack = cms.EDProducer("L1TRawToDigi",
        Setup           = cms.string("stage2::EMTFSetup"),
        InputLabel      = cms.InputTag("rawDataCollector"),
        FedIds          = cms.vint32( 1384, 1385 ),
        FWId            = cms.uint32(0),
        debug = cms.untracked.bool(False), ## More debugging output
        MTF7 = cms.untracked.bool(True)
)


# process.out = cms.OutputModule("PoolOutputModule", 
#    outputCommands=cms.untracked.vstring(
#        'keep *_unpack_*_*',
#        'keep *_*_*_EMTF',
#        'keep *l1t*_*_*_*',
#        'keep recoMuons_muons__RECO',
#        'keep *_*osmic*_*_*',
#        'keep edmTriggerResults_*_*_*',
#        'keep *CSC*_*_*_*',
#        'keep *_*csc*_*_*',
#        'keep *_*_*csc*_*',
#    ),
#    fileName = cms.untracked.string("EMTF_RAWToRoot_v0.root")
# )

process.out = cms.OutputModule("PoolOutputModule", 
   outputCommands=cms.untracked.vstring(
       'keep *_unpack_*_*',
       'keep *_*_*_EMTF',
       'keep *l1t*_*_*_*',
       'keep recoMuons_muons__RECO',
       'keep *_*osmic*_*_*',
       'keep edmTriggerResults_*_*_*',
       'keep *CSC*_*_*_*',
       'keep *_*CSC*_*_*',
       'keep *_*_*CSC*_*',
       'keep *_*_*_*CSC*',
       'keep *Csc*_*_*_*',
       'keep *_*Csc*_*_*',
       'keep *_*_*Csc*_*',
       'keep *_*_*_*Csc*',
       'keep *csc*_*_*_*',
       'keep *_*csc*_*_*',
       'keep *_*_*csc*_*',
       'keep *_*_*_*csc*',
   ),
   fileName = cms.untracked.string("EMTF_RAWToRoot_v0.root")
)

## process.p = cms.Path(process.dump * process.unpack)
process.p = cms.Path(process.unpack)
process.end = cms.EndPath(process.out)
