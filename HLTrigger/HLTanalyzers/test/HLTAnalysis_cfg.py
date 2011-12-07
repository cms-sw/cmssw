import FWCore.ParameterSet.Config as cms

##################################################################

# useful options
isData=1 # =1 running on real data, =0 running on MC


OUTPUT_HIST='openhlt.root'
NEVTS=100
MENU="GRun" # GRun for data or MC with >= CMSSW_3_8_X
isRelval=0 # =0 for running on MC RelVals, =0 for standard production MC, no effect for data 

isRaw=1 #  =1 changes the path to run lumiProducer from dbs on RAW, =0 if RAW+RECO will get lumi info from file intself


#WhichHLTProcess="HLT"
WhichHLTProcess="HLT"

####   MC cross section weights in pb, use 1 for real data  ##########

XS_7TeV_MinBias=7.126E10  # from Summer09 production
XS_10TeV_MinBias=7.528E10
XS_900GeV_MinBias=5.241E10
XSECTION=1.
FILTEREFF=1.              # gen filter efficiency

if (isData):
    XSECTION=1.         # cross section weight in pb
    FILTEREFF=1.
    MENU="GRun"

#####  Global Tag ###############################################
    
# Which AlCa condition for what. 

if (isData):
    GLOBAL_TAG='GR_R_50_V3::All' # 2011 Collisions data, CMSSW_5_0_X
else:
    GLOBAL_TAG='START42_V12::All' # CMSSW_4_2_X MC, STARTUP Conditions
    
##################################################################
import os
cmsswVersion = os.environ['CMSSW_VERSION']

process = cms.Process("ANALYSIS")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.suppressWarning = cms.untracked.vstring( 'hltOnlineBeamSpot' )
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False)
)


## For running on RAW only 
process.source = cms.Source("PoolSource",
                             fileNames = cms.untracked.vstring(
<<<<<<< HLTAnalysis_cfg.py
    '/store/data/Run2011B/DoubleMu/RAW/v1/000/178/479/2875F9DE-1AF6-E011-94BC-001D09F2983F.root'
=======
                            '/store/data/Run2011B/DoubleMu/RAW/v1/000/176/304/9A59C03B-C2DE-E011-A854-BCAEC53296F3.root'
                                 )
>>>>>>> 1.56
                                 )
                            )


## For running on RAW+RECO
##process.source = cms.Source("PoolSource",
##  fileNames = cms.untracked.vstring(
##
##   ),
##  secondaryFileNames =  cms.untracked.vstring(
##
##  )
##)

<<<<<<< HLTAnalysis_cfg.py

=======
# from CMSSW_5_0_0_pre6: RawDataLikeMC=False (to keep "source")
if cmsswVersion > "CMSSW_5_0":
    process.source.labelRawDataLikeMC = cms.untracked.bool( False )

>>>>>>> 1.56
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( NEVTS ),
    skipBadFiles = cms.bool(True)
    )

<<<<<<< HLTAnalysis_cfg.py
process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
=======
process.load("HLTrigger.HLTanalyzers.HLT_ES_cff")
>>>>>>> 1.56
process.GlobalTag.globaltag = GLOBAL_TAG
process.GlobalTag.connect   = 'frontier://FrontierProd/CMS_COND_31X_GLOBALTAG'
process.GlobalTag.pfnPrefix = cms.untracked.string('frontier://FrontierProd/')

process.load('Configuration/StandardSequences/SimL1Emulator_cff')

# OpenHLT specificss
# Define the HLT reco paths
process.load("HLTrigger.HLTanalyzers.HLTopen_cff")

process.DQM = cms.Service( "DQM",)
process.DQMStore = cms.Service( "DQMStore",)

# Define the analyzer modules
process.load("HLTrigger.HLTanalyzers.HLTAnalyser_cfi")
if (isRaw):
    process.analyzeThis = cms.Path(process.lumiProducer + process.HLTBeginSequence + process.hltanalysis )   
else:
    process.analyzeThis = cms.Path( process.HLTBeginSequence + process.hltanalysis )





process.hltanalysis.RunParameters.HistogramFile=OUTPUT_HIST
process.hltanalysis.xSection=XSECTION
process.hltanalysis.filterEff=FILTEREFF
process.hltanalysis.l1GtReadoutRecord = cms.InputTag( 'hltGtDigis','',process.name_() ) # get gtDigis extract from the RAW
process.hltanalysis.hltresults = cms.InputTag( 'TriggerResults','',WhichHLTProcess)
process.hltanalysis.HLTProcessName = WhichHLTProcess
process.hltanalysis.ht = "hltJet40Ht"
process.hltanalysis.genmet = "genMetTrue"

# Switch on ECAL alignment to be consistent with full HLT Event Setup
#process.EcalBarrelGeometryEP.applyAlignment = True
#process.EcalEndcapGeometryEP.applyAlignment = True
#process.EcalPreshowerGeometryEP.applyAlignment = True

# Add tight isolation PF taus
process.HLTPFTauSequence += process.hltPFTausTightIso

if (MENU == "GRun"):
    # get the objects associated with the menu
    process.hltanalysis.IsoPixelTracksL3 = "hltHITIPTCorrector8E29"
    process.hltanalysis.IsoPixelTracksL2 = "hltIsolPixelTrackProd8E29"
    if (isData == 0):
        if(isRelval == 0):
            process.hltTrigReport.HLTriggerResults = "TriggerResults::HLT"
            process.hltanalysis.l1GtObjectMapRecord = "hltL1GtObjectMap::HLT"
            process.hltanalysis.hltresults = "TriggerResults::HLT"
        else:
            process.hltTrigReport.HLTriggerResults = "TriggerResults::HLT"
            process.hltanalysis.l1GtObjectMapRecord = "hltL1GtObjectMap::HLT"
            process.hltanalysis.hltresults = "TriggerResults::HLT"
                                                                                                            
# pdt, if running on MC
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")


# Schedule the whole thing
if (MENU == "GRun"):
    process.schedule = cms.Schedule(
        process.DoHLTJets,
        process.DoHltMuon,
        process.DoHLTPhoton,
        process.DoHLTElectron,
        process.DoHLTTau,
        process.DoHLTBTag,
        process.DoHLTMinBiasPixelTracks,
        process.analyzeThis
        )
        
#########################################################################################
#
if (isData):  # replace all instances of "rawDataCollector" with "source" in InputTags
    from FWCore.ParameterSet import Mixins
    for module in process.__dict__.itervalues():
        if isinstance(module, Mixins._Parameterizable):
            for parameter in module.__dict__.itervalues():
                if isinstance(parameter, cms.InputTag):
                    if parameter.moduleLabel == 'rawDataCollector':
                        parameter.moduleLabel = 'source'
                
else:
    if (MENU == "GRun"):
        from FWCore.ParameterSet import Mixins
        for module in process.__dict__.itervalues():
            if isinstance(module, Mixins._Parameterizable):
                for parameter in module.__dict__.itervalues():
                    if isinstance(parameter, cms.InputTag):
                        if parameter.moduleLabel == 'rawDataCollector':
                            if(isRelval == 0):
                                parameter.moduleLabel = 'rawDataCollector::HLT'


