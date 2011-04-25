import FWCore.ParameterSet.Config as cms

##################################################################

# useful options
isData=1 # =1 running on real data, =0 running on MC


OUTPUT_HIST='openhlt.root'
NEVTS=200
MENU="GRun" # GRun for data or MC with >= CMSSW_3_8_X
isRelval=0 # =0 for running on MC RelVals, =0 for standard production MC, no effect for data 

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
    # GLOBAL_TAG='GR09_H_V6OFF::All' # collisions 2009
    # GLOBAL_TAG='GR10_H_V6A::All' # collisions2010 tag for CMSSW_3_6_X
    # GLOBAL_TAG='GR10_H_V8_T2::All' # collisions2010 tag for CMSSW_3_8_X
    # GLOBAL_TAG='GR10_H_V9::All' # collisions2010 tag for CMSSW_3_8_X, updated  
    # GLOBAL_TAG='GR_R_311_V0::All' # Temporary tag for running in CMSSW_3_11_X
##    GLOBAL_TAG='L1HLTST311_V0::All'
    ## Use the same GLOBAL TAG as in the master table
#    GLOBAL_TAG='TESTL1_GR_P::All'    
    GLOBAL_TAG='GR_H_V15::All'
else:
    GLOBAL_TAG='START311_V2::All'
    if (MENU == "GRun"): GLOBAL_TAG= 'START311_V2::All'
    
##################################################################

process = cms.Process("ANALYSIS")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.suppressWarning = cms.untracked.vstring( 'hltOnlineBeamSpot' )
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False)
)

## process.source = cms.Source("PoolSource",
##     fileNames = cms.untracked.vstring(
##                                 '/store/data/Run2011A/SingleMu/RAW/v1/000/160/406/CA9EFACF-A14D-E011-ACB9-00304879EDEA.root'
##     )
## )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/data/Run2011A/Jet/RECO/PromptReco-v1/000/161/103/9CE9D4FC-AB56-E011-BB74-0030487CD6E6.root',
    '/store/data/Run2011A/Jet/RECO/PromptReco-v1/000/161/103/9802528B-A456-E011-9133-0030487CAF5E.root',
    '/store/data/Run2011A/Jet/RECO/PromptReco-v1/000/161/103/10F88E31-9F56-E011-9AEB-0030487CD14E.root'
     ),
    secondaryFileNames =  cms.untracked.vstring(
    '/store/data/Run2011A/Jet/RAW/v1/000/161/103/E26D5DA4-8654-E011-A83F-001D09F28D4A.root',
    '/store/data/Run2011A/Jet/RAW/v1/000/161/103/D68B264D-8054-E011-A7B2-001617E30D0A.root',
    '/store/data/Run2011A/Jet/RAW/v1/000/161/103/8654BB88-8254-E011-B1A1-001617C3B65A.root'
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

# Remove the PrescaleService which, in 31X, it is expected once HLT_XXX_cff is imported
# del process.PrescaleService ## ccla no longer needed in for releases in 33x+?

process.DQM = cms.Service( "DQM",)
process.DQMStore = cms.Service( "DQMStore",)

#offline vertices with deterministic annealing. Should become the default as of 4_2_0_pre7. Requires > V01-04-04      RecoVertex/PrimaryVertexProducer
process.load("RecoVertex.Configuration.RecoVertex_cff")
from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi import *
process.load("RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi")
process.offlinePrimaryVerticesDA = process.offlinePrimaryVertices.clone()

# Define the analyzer modules
process.load("HLTrigger.HLTanalyzers.HLTAnalyser_cfi")
process.analyzeThis = cms.Path( process.offlinePrimaryVerticesDA + process.HLTBeginSequence + process.hltanalysis )

process.hltanalysis.RunParameters.HistogramFile=OUTPUT_HIST
process.hltanalysis.xSection=XSECTION
process.hltanalysis.filterEff=FILTEREFF
process.hltanalysis.l1GtReadoutRecord = cms.InputTag( 'hltGtDigis','',process.name_() ) # get gtDigis extract from the RAW
process.hltanalysis.hltresults = cms.InputTag( 'TriggerResults','',WhichHLTProcess)
process.hltanalysis.HLTProcessName = WhichHLTProcess
process.hltanalysis.ht = "hltJet30Ht"
process.hltanalysis.genmet = "genMetTrue"
process.hltanalysis.PrimaryVerticesHLT = cms.InputTag('pixelVertices')
process.hltanalysis.OfflinePrimaryVertices0 = cms.InputTag('offlinePrimaryVertices')
process.hltanalysis.OfflinePrimaryVertices1 = cms.InputTag('offlinePrimaryVerticesDA')



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
                                                                                                            
# pdt
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
        process.analyzeThis)
        
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
