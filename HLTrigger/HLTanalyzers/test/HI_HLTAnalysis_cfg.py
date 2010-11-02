import FWCore.ParameterSet.Config as cms

##################################################################

# useful options
isData=0 # =1 running on real data, =0 running on MC


OUTPUT_HIST='openhlt.root'
NEVTS=-1
MENU="HIon" # LUMI8e29 or LUMI1e31 for pre-38X MC, or GRun for data
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
    GLOBAL_TAG='GR10_H_V8_T2::All' # collisions2010 tag for CMSSW_3_8_X
else:
    GLOBAL_TAG='MC_31X_V2::All'
    if (MENU == "LUMI8e29"): GLOBAL_TAG= 'STARTUP3X_V15::All'
    if (MENU == "HIon"): GLOBAL_TAG= 'START39_V3::All'
    
    
##################################################################

process = cms.Process("ANALYSIS")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    'dcache:/pnfs/cmsaf.mit.edu/t2bat/cms/store/user/nart/Hydjet_Quenched_MinBias_2760GeV_STARTUP39V3_L1Menu_CollisionsHeavyIons2010_v0_391v1/Hydjet_Quenched_MinBias_2760GeV_STARTUP39V3_L1Menu_CollisionsHeavyIons2010_v0_391v1_RECO/4fa2e3ec03e211b495652e2ed26839f0/output_9_1_mJS.root'
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( NEVTS ),
    skipBadFiles = cms.bool(True)
    )

process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('Configuration/StandardSequences/SimL1Emulator_cff')
process.GlobalTag.globaltag = GLOBAL_TAG
process.GlobalTag.toGet = cms.VPSet(
    cms.PSet(record = cms.string("HeavyIonRcd"),
             tag = cms.string("CentralityTable_HFhits40_Hydjet2760GeV_v0_mc"),
             connect = cms.untracked.string("frontier://FrontierPrep/CMS_COND_PHYSICSTOOLS")
             )
    )


# OpenHLT specificss
# Define the HLT reco paths
process.load("HLTrigger.HLTanalyzers.HI_HLTopen_cff")

# Remove the PrescaleService which, in 31X, it is expected once HLT_XXX_cff is imported
# del process.PrescaleService ## ccla no longer needed in for releases in 33x+?

process.DQM = cms.Service( "DQM",)
process.DQMStore = cms.Service( "DQMStore",)

# AlCa OpenHLT specific settings

# Define the analyzer modules
process.load("HLTrigger.HLTanalyzers.HI_HLTAnalyser_cff")
process.analyzeThis = cms.Path( process.HLTBeginSequence 
    * process.heavyIon
    * process.siPixelRecHits
    * process.hiCentrality
    * process.centralityBin
    * process.hltanalysis
    )

process.hltanalysis.RunParameters = cms.PSet(
	    HistogramFile	  = cms.untracked.string(OUTPUT_HIST),
	    UseTFileService	  = cms.untracked.bool(True),
            Monte                = cms.bool(True),
            Debug                = cms.bool(False),

                ### added in 2010 ###
            DoHeavyIon           = cms.untracked.bool(True),
            DoMC           = cms.untracked.bool(True),
            DoAlCa           = cms.untracked.bool(True),
            DoTracks           = cms.untracked.bool(True),
            DoVertex           = cms.untracked.bool(True),
            DoJets           = cms.untracked.bool(True),

                ### MCTruth
            DoParticles          = cms.untracked.bool(False),
            DoRapidity           = cms.untracked.bool(False),
            DoVerticesByParticle = cms.untracked.bool(False),

                ### Egamma
            DoPhotons            = cms.untracked.bool(True),
            DoElectrons          = cms.untracked.bool(False),
            DoSuperClusters      = cms.untracked.bool(True),

                ### Muon
            DoMuons            = cms.untracked.bool(True),
            DoL1Muons            = cms.untracked.bool(True),
            DoL2Muons            = cms.untracked.bool(False),
            DoL3Muons            = cms.untracked.bool(False),
            DoOfflineMuons       = cms.untracked.bool(False),
            DoQuarkonia          = cms.untracked.bool(False)
            )

process.hltanalysis.xSection=XSECTION
process.hltanalysis.filterEff=FILTEREFF
process.hltanalysis.l1GtReadoutRecord = cms.InputTag( 'hltGtDigis','',process.name_() ) # get gtDigis extract from the RAW
process.hltanalysis.l1GtObjectMapRecord = cms.InputTag("hltL1GtObjectMap","",WhichHLTProcess)
process.hltanalysis.hltresults = cms.InputTag( 'TriggerResults','',WhichHLTProcess)
process.hltanalysis.HLTProcessName = WhichHLTProcess
process.hltanalysis.ht = "hltJet30Ht"
process.hltanalysis.genmet = "genMetTrue"

# TFile service output
process.TFileService = cms.Service('TFileService',
    fileName = cms.string("hltana.root")
    )

# Schedule the whole thing
if (MENU == "HIon"):
    print "menu HIon"
    process.schedule = cms.Schedule(
	process.DoHLTHIJets,
	process.DoHLTHIPhoton,
	process.analyzeThis)

from L1Trigger.Configuration.L1Trigger_custom import customiseL1Menu
process=customiseL1Menu(process)
process.l1conddb.connect = "sqlite_file:./L1Menu_CollisionsHeavyIons2010_v0_mc.db"

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
