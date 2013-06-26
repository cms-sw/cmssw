import FWCore.ParameterSet.Config as cms

##################################################################

# useful options
isData=1 # =1 running on real data, =0 running on MC


OUTPUT_HIST='openhlt.root'
NEVTS=100
MENU="GRun" # GRun for data or MC with >= CMSSW_3_8_X
isRelval=0 # =0 for running on MC RelVals, =0 for standard production MC, no effect for data 

isRaw=0 #  =1 changes the path to run lumiProducer from dbs on RAW, =0 if RAW+RECO will get lumi info from file intself
        #  Note - rerunning the lumiProducer puts a heavy load on the DB and should be used only if really needed.  


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

import os
cmsswVersion = os.environ['CMSSW_VERSION']
    
# Which AlCa condition for what. 
if cmsswVersion > "CMSSW_5_0" :
    if (isData) :
        GLOBAL_TAG='GR_R_52_V1::All' # 2011 Collisions data, CMSSW_5_0_X
    else :
        GLOBAL_TAG='START52_V1::All'

else : # == if 44X
    if (isData):
        GLOBAL_TAG='GR_R_44_V11::All' # 2011 Collisions data, CMSSW_5_0_X
    else:
        GLOBAL_TAG='START44_V5::All' # CMSSW_4_2_X MC, STARTUP Conditions
    
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
    '/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/041B0376-FA02-E111-B0C0-003048F1110E.root'
   #'/store/relval/CMSSW_4_4_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START44_V5-v2/0044/063F2082-D2E5-E011-99C8-00248C0BE018.root'
   #'/store/data/Run2011B/ZeroBiasHPF2/RAW/v1/000/179/828/8ED5066B-43FF-E011-BCD2-003048F024FE.root'
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

# from CMSSW_5_0_0_pre6: RawDataLikeMC=False (to keep "source")
if cmsswVersion > "CMSSW_5_0":
    process.source.labelRawDataLikeMC = cms.untracked.bool( False )


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( NEVTS ),
    skipBadFiles = cms.bool(True)
    )

##LUMI CODE
if(isRaw):
    from RecoLuminosity.LumiProducer.lumiProducer_cff import *
    process.load('RecoLuminosity.LumiProducer.lumiProducer_cff')

process.load("HLTrigger.HLTanalyzers.HLT_ES_cff")

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
##LUMI CODE
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

####### L1 jets - probably would be cleaner to move this to an external config file...
process.hltGctDigis.hltMode = cms.bool( False )

# to run the emulator on the output of the unpacker (which we run as part of HLTBeginSequence, independant of the emulator per se)
process.load('L1Trigger.GlobalCaloTrigger.gctDigis_cfi')
process.gctDigis.writeInternalData = cms.bool(True)
process.gctDigis.inputLabel = cms.InputTag("hltGctDigis")

# Create the uncorrected intermediate jets
process.load("EventFilter.GctRawToDigi.gctInternJetProd_cfi")
process.gctInternJetProducer.internalJetSource = cms.InputTag("gctDigis")
process.L1Jets = cms.Path(process.gctDigis + process.gctInternJetProducer)


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

## may be useful for debugging purposes
## process.out = cms.OutputModule("PoolOutputModule",
##                              fileName = cms.untracked.string('test.root'),
##                               #SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('FilterPath')),
##                              outputCommands = cms.untracked.vstring('keep *_hltParticleFlow_*_*',
##                                                                     'keep *_hltL1extraParticles_*_*',
##                                                                     'keep *_whatever_*_*'
##                                                                     )
##                             )
## process.outpath = cms.EndPath(process.out)



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
        process.L1Jets,
        process.analyzeThis## ,
##         process.outpath
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


