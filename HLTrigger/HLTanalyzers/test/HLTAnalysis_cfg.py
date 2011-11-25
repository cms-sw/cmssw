import FWCore.ParameterSet.Config as cms

##################################################################

# useful options
isData=1 # =1 running on real data, =0 running on MC


OUTPUT_HIST='openhlt.root'
NEVTS=-1
MENU="GRun" # GRun for data or MC with >= CMSSW_3_8_X
isRelval=0 # =0 for running on MC RelVals, =0 for standard production MC, no effect for data 

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
    GLOBAL_TAG='GR_H_V22::All' # 2011 Collisions data, CMSSW_4_2_X
else:
    GLOBAL_TAG='START42_V12::All' # CMSSW_4_2_X MC, STARTUP Conditions
    
##################################################################

process = cms.Process("ANALYSIS")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.suppressWarning = cms.untracked.vstring( 'hltOnlineBeamSpot' )
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False)
)

## For running on RAW only 
## process.source = cms.Source("PoolSource",
##                             fileNames = cms.untracked.vstring(
## '/store/data/Run2011B/SingleMu/RAW-RECO/WMu-PromptSkim-v1/0000/005BD2DB-E6FF-E011-BFF0-78E7D1E49B52.root'
##   #  '/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/16BCCB82-0303-E111-AB2A-BCAEC5364C4B.root'
##                                 #'/store/data/Run2011A/MultiJet/RAW/v1/000/173/243/E2ACC2F2-5EC5-E011-964F-003048F118AA.root'
##                                 )
##                                 )

## For running on RAW+RECO
process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
    '/store/data/Run2011B/SingleMu/AOD/PromptReco-v1/000/180/250/002BB011-F904-E111-AFCC-BCAEC518FF63.root',
   ),
  secondaryFileNames =  cms.untracked.vstring(
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/041B0376-FA02-E111-B0C0-003048F1110E.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/0A7987BC-0C03-E111-A1AD-001D09F24399.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/0A7A8859-F502-E111-A3E7-BCAEC53296F7.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/142979C5-0203-E111-9247-E0CB4E4408E3.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/16BCCB82-0303-E111-AB2A-BCAEC5364C4B.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/289F6D8F-0503-E111-B8EA-003048D3C932.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/28FF0FF9-F302-E111-8245-001D09F2432B.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/30AEA875-FA02-E111-859E-BCAEC53296F7.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/30B3D5C6-F402-E111-9793-BCAEC54DB5D6.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/3201E336-FF02-E111-BF31-BCAEC532970A.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/3AEC6635-F802-E111-982E-001D09F251D1.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/3CDE53EB-0B03-E111-8E1C-003048D37538.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/44E1F048-0603-E111-BBA3-E0CB4E4408E3.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/482D08F9-F302-E111-965D-001D09F23A34.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/58A4FD09-0203-E111-ABAC-BCAEC5364C4B.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/6885BAC4-0203-E111-A07F-BCAEC5364C4B.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/6A978CC5-0203-E111-9E94-BCAEC5364C4C.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/6C9E7160-0103-E111-9282-BCAEC518FF76.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/6CD4CD01-FB02-E111-9016-BCAEC518FF3C.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/76603BBC-FB02-E111-B183-BCAEC532970D.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/78723316-0903-E111-B849-BCAEC5329727.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/7C58E9D1-0703-E111-B659-0025901D631E.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/7EE1E882-0303-E111-B3C7-BCAEC532972C.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/82DF0CEB-0B03-E111-A348-003048678110.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/82E41517-0903-E111-96B7-E0CB4E4408D5.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/8A67617A-F702-E111-A868-0019B9F730D2.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/8E3600DF-FD02-E111-85BF-BCAEC532972D.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/9222EE48-0603-E111-8EDA-BCAEC518FF8F.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/A0F801F0-F802-E111-B952-0025901D624E.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/A20CC381-0303-E111-81A4-003048D2BA82.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/A2D8938F-FE02-E111-8D19-0025B32445E0.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/A61C9F6B-FC02-E111-8423-003048D2C1C4.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/A61F77BE-FB02-E111-82CF-0025901D627C.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/A6519A06-1303-E111-A12E-001D09F2516D.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/B4ADC235-0B03-E111-973A-001D09F295A1.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/B83EA431-F802-E111-8974-001D09F28E80.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/BA8D372A-F802-E111-9144-001D09F24493.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/C24DD7C5-F402-E111-BC50-BCAEC532970E.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/C2D9C96C-FC02-E111-8CB9-003048D37560.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/D0DE5FC5-0903-E111-83A9-BCAEC5329707.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/D4178559-F502-E111-8D95-E0CB4E4408C4.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/D435AF6C-0803-E111-8DAF-BCAEC518FF41.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/D459276A-0F03-E111-A215-BCAEC518FF80.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/D656421A-0E03-E111-A2D7-BCAEC5329707.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/DAF54B7A-F702-E111-86EC-001D09F2841C.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/DE03A174-FA02-E111-93E6-E0CB4E553673.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/DE1E176C-0803-E111-97D5-BCAEC5329721.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/E69CE8BA-0C03-E111-A1E1-001D09F2525D.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/EA018CED-FF02-E111-982B-0025B32445E0.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/F6C648DD-0403-E111-9B79-003048D37560.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/FC68AF93-FE02-E111-A7B8-001D09F2437B.root',
'/store/data/Run2011B/SingleMu/RAW/v1/000/180/250/FE27384F-0D03-E111-9D97-003048F118C6.root'


  )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( NEVTS ),
    skipBadFiles = cms.bool(True)
    )

process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
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
process.EcalBarrelGeometryEP.applyAlignment = True
process.EcalEndcapGeometryEP.applyAlignment = True
process.EcalPreshowerGeometryEP.applyAlignment = True

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


process.out = cms.OutputModule("PoolOutputModule",
                              fileName = cms.untracked.string('test.root'),
                               #SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('FilterPath')),
                              outputCommands = cms.untracked.vstring('keep *_hltParticleFlow_*_*',
                                                                     'keep *_pfAllMuons_*_*',
                                                                     'keep *_muons_*_RECO')

                              )


process.outpath = cms.EndPath(process.out)




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
        process.analyzeThis,
        process.outpath
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





