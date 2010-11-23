import FWCore.ParameterSet.Config as cms

##################################################################

# useful options
isData=1 # =1 running on real data, =0 running on MC


OUTPUT_HIST='openhlt.root'
NEVTS=500
MENU="LUMI8e29" # LUMI8e29 or LUMI1e31 for pre-38X MC, or GRun for data
isRelval=1 # =1 for running on MC RelVals, =0 for standard production MC, no effect for data 

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
    
# Which AlCa condition for what. Available from pre11
# * DESIGN_31X_V1 - no smearing, alignment and calibration constants = 1.  No bad channels.
# * MC_31X_V1 (was IDEAL_31X) - conditions intended for 31X physics MC production: no smearing,
#   alignment and calibration constants = 1.  Bad channels are masked.
# * STARTUP_31X_V1 (was STARTUP_31X) - conditions needed for HLT 8E29 menu studies: As MC_31X_V1 (including bad channels),
#   but with alignment and calibration constants smeared according to knowledge from CRAFT.
# * CRAFT08_31X_V1 (was CRAFT_31X) - conditions for CRAFT08 reprocessing.
# * CRAFT_31X_V1P, CRAFT_31X_V1H - initial conditions for 2009 cosmic data taking - as CRAFT08_31X_V1 but with different
#   tag names to allow append IOV, and DT cabling map corresponding to 2009 configuration (10 FEDs).
# Meanwhile...:

if (isData):
    # GLOBAL_TAG='GR09_H_V6OFF::All' # collisions 2009
    # GLOBAL_TAG='GR10_H_V6A::All' # collisions2010 tag for CMSSW_3_6_X
    # GLOBAL_TAG='GR10_H_V8_T2::All' # collisions2010 tag for CMSSW_3_8_X
    GLOBAL_TAG='GR10_H_V12::All' # Temporary tag for CMSSW_3_10_X
else:
    GLOBAL_TAG='MC_31X_V2::All'
    if (MENU == "LUMI8e29"): GLOBAL_TAG= 'STARTUP3X_V15::All'
    
    
##################################################################

process = cms.Process("ANALYSIS")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
                                '/store/data/Run2010B/MinimumBias/RAW/v1/000/146/511/F43D188C-7BC7-DF11-B1C8-00304879EE3E.root',
                                '/store/data/Run2010B/MinimumBias/RAW/v1/000/146/511/D682093A-56C7-DF11-A939-001617DBD5AC.root',
                                '/store/data/Run2010B/MinimumBias/RAW/v1/000/146/511/968DEC22-5BC7-DF11-87F5-0030487CAEAC.root',
                                '/store/data/Run2010B/MinimumBias/RAW/v1/000/146/511/94C04FB2-78C7-DF11-86E4-003048D2BE08.root',
                                '/store/data/Run2010B/MinimumBias/RAW/v1/000/146/511/8CEC7EBC-7FC7-DF11-9ED2-001D09F24493.root',
                                '/store/data/Run2010B/MinimumBias/RAW/v1/000/146/511/86044E5C-77C7-DF11-902B-001D09F28F25.root',
                                '/store/data/Run2010B/MinimumBias/RAW/v1/000/146/511/6EC49446-5DC7-DF11-B529-001D09F2447F.root',
                                '/store/data/Run2010B/MinimumBias/RAW/v1/000/146/511/626A9195-63C7-DF11-8640-001617E30F58.root',
                                '/store/data/Run2010B/MinimumBias/RAW/v1/000/146/511/60C3055C-72C7-DF11-946C-003048D37560.root',
                                '/store/data/Run2010B/MinimumBias/RAW/v1/000/146/511/608BFC4B-69C7-DF11-95C9-0015C5FDE067.root',
                                '/store/data/Run2010B/MinimumBias/RAW/v1/000/146/511/5EEA6071-5FC7-DF11-8699-003048D2BC42.root',
                                '/store/data/Run2010B/MinimumBias/RAW/v1/000/146/511/426B8B67-66C7-DF11-A95D-000423D98B6C.root',
                                '/store/data/Run2010B/MinimumBias/RAW/v1/000/146/511/3A168679-61C7-DF11-AAA2-003048F11114.root',
                                '/store/data/Run2010B/MinimumBias/RAW/v1/000/146/511/28AD7271-6BC7-DF11-B09E-0030487C7828.root',
                                '/store/data/Run2010B/MinimumBias/RAW/v1/000/146/511/2801608B-6FC7-DF11-A582-001D09F2B2CF.root',
                                '/store/data/Run2010B/MinimumBias/RAW/v1/000/146/511/1AC021D5-6CC7-DF11-861D-001D09F2924F.root',
                                '/store/data/Run2010B/MinimumBias/RAW/v1/000/146/511/16620D83-58C7-DF11-A0B8-001D09F23A20.root',
                                '/store/data/Run2010B/MinimumBias/RAW/v1/000/146/511/1218758E-74C7-DF11-B72D-001D09F24FEC.root',
                                '/store/data/Run2010B/MinimumBias/RAW/v1/000/146/511/0813DBF5-83C7-DF11-A2C3-003048F118C6.root'
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

# AlCa OpenHLT specific settings

# Define the analyzer modules
process.load("HLTrigger.HLTanalyzers.HLTAnalyser_cfi")
process.analyzeThis = cms.Path( process.HLTBeginSequence + process.hltanalysis )

process.hltanalysis.RunParameters.HistogramFile=OUTPUT_HIST
process.hltanalysis.xSection=XSECTION
process.hltanalysis.filterEff=FILTEREFF
process.hltanalysis.l1GtReadoutRecord = cms.InputTag( 'hltGtDigis','',process.name_() ) # get gtDigis extract from the RAW
process.hltanalysis.hltresults = cms.InputTag( 'TriggerResults','',WhichHLTProcess)
process.hltanalysis.HLTProcessName = WhichHLTProcess
process.hltanalysis.ht = "hltJet30Ht"
process.hltanalysis.genmet = "genMetTrue"

if (MENU == "GRun"):
    # get the objects associated with the 8e29 menu
    process.hltanalysis.recjets = "hltMCJetCorJetIcone5HF07"
    process.hltanalysis.ht = "hltJet20UHt"
    process.hltanalysis.IsoPixelTracksL3 = "hltHITIPTCorrector8E29"
    process.hltanalysis.IsoPixelTracksL2 = "hltIsolPixelTrackProd8E29"
    if (isData == 0):
        if(isRelval == 0):
            process.hltTrigReport.HLTriggerResults = "TriggerResults::HLT8E29"
            process.hltanalysis.l1GtObjectMapRecord = "hltL1GtObjectMap::HLT8E29"
            process.hltanalysis.hltresults = "TriggerResults::HLT8E29"
        else:
            process.hltTrigReport.HLTriggerResults = "TriggerResults::HLT"
            process.hltanalysis.l1GtObjectMapRecord = "hltL1GtObjectMap::HLT"
            process.hltanalysis.hltresults = "TriggerResults::HLT"
                                                                                                            
elif (MENU == "LUMI8e29"):
    # get the objects associated with the 8e29 menu
    process.hltanalysis.recjets = "hltMCJetCorJetIcone5HF07"    
    process.hltanalysis.ht = "hltJet20UHt"    
    process.hltanalysis.IsoPixelTracksL3 = "hltHITIPTCorrector8E29"
    process.hltanalysis.IsoPixelTracksL2 = "hltIsolPixelTrackProd8E29"
    if (isData == 0):
        if(isRelval == 0):
            process.hltTrigReport.HLTriggerResults = "TriggerResults::HLT8E29"
            process.hltanalysis.l1GtObjectMapRecord = "hltL1GtObjectMap::HLT8E29"
            process.hltanalysis.hltresults = "TriggerResults::HLT8E29"
        else:
            process.hltTrigReport.HLTriggerResults = "TriggerResults::HLT"
            process.hltanalysis.l1GtObjectMapRecord = "hltL1GtObjectMap::HLT"
            process.hltanalysis.hltresults = "TriggerResults::HLT"

# pdt
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

# Schedule the whole thing
if (MENU == "GRun"):
    process.schedule = cms.Schedule(
        process.DoHLTJetsU,
        process.DoHltMuon,
        process.DoHLTPhoton,
        process.DoHLTElectron,
        process.DoHLTTau,
        process.DoHLTBTag,
        process.DoHLTAlCaECALPhiSym,
        process.DoHLTAlCaPi0Eta8E29,
        process.DoHLTMinBiasPixelTracks,
        process.analyzeThis)
        
elif (MENU == "LUMI8e29"):
    process.schedule = cms.Schedule(
        process.DoHLTJetsU,
        process.DoHltMuon,
        process.DoHLTPhoton,
        process.DoHLTElectron,
        ##        process.DoHLTElectronStartUpWindows,
        ##        process.DoHLTElectronLargeWindows,
        ##        process.DoHLTElectronSiStrip,
        process.DoHLTTau,
        process.DoHLTBTag,
        process.DoHLTAlCaECALPhiSym,
        process.DoHLTAlCaPi0Eta8E29,
        # process.DoHLTIsoTrack8E29, 
        process.DoHLTMinBiasPixelTracks,
        process.analyzeThis)
else:
    process.schedule = cms.Schedule( 
        process.DoHLTJets, 
        process.DoHltMuon, 
        process.DoHLTPhoton, 
        process.DoHLTElectron, 
        ##        process.DoHLTElectronStartUpWindows, 
        ##        process.DoHLTElectronLargeWindows,
        ##        process.DoHLTElectronSiStrip,
        process.DoHLTTau, 
        process.DoHLTBTag,
        process.DoHLTAlCaECALPhiSym,
        process.DoHLTAlCaPi0Eta1E31,
        # process.DoHLTIsoTrack,
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
    if (MENU == "LUMI8e29"):
        from FWCore.ParameterSet import Mixins
        for module in process.__dict__.itervalues():
            if isinstance(module, Mixins._Parameterizable):
                for parameter in module.__dict__.itervalues():
                    if isinstance(parameter, cms.InputTag):
                        if parameter.moduleLabel == 'rawDataCollector':
                            if(isRelval == 0):
                                parameter.moduleLabel = 'rawDataCollector::HLT8E29'
