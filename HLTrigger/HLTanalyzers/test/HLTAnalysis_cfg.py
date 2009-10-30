import FWCore.ParameterSet.Config as cms

##################################################################

OUTPUT_HIST='openhlt.root'
NEVTS=100
XSECTION=7.126E10         # cross section weight in pb
FILTEREFF=1.              # gen filter efficiency

##################################################################

process = cms.Process("ANALYSIS")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/mc/Summer09/MinBias900GeV/GEN-SIM-RAW/MC_31X_V3-v1/0029/EA9124F5-A28C-DE11-B015-0016367B47AF.root'
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( NEVTS ),
    skipBadFiles = cms.bool(True)
    )

process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
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
process.GlobalTag.globaltag = 'MC_31X_V2::All'


process.load('Configuration/StandardSequences/SimL1Emulator_cff')

# OpenHLT specificss
# Define the HLT reco paths
process.load("HLTrigger.HLTanalyzers.HLTopen_cff")
# Remove the PrescaleService which, in 31X, it is expected once HLT_XXX_cff is imported
del process.PrescaleService

# AlCa OpenHLT specific settings

# Define the analyzer modules
process.load("HLTrigger.HLTanalyzers.HLTAnalyser_cfi")
process.analyzeThis = cms.Path( process.hltanalysis )
process.hltanalysis.RunParameters.HistogramFile=OUTPUT_HIST
process.hltanalysis.xSection=XSECTION
process.hltanalysis.filterEff=FILTEREFF

# pdt
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

# Schedule the whole thing
process.schedule = cms.Schedule( 
    process.DoHLTJets, 
    process.DoHltMuon, 
    process.DoHLTPhoton, 
##    process.DoHLTElectron, 
    process.DoHLTElectronStartUpWindows, 
    process.DoHLTElectronLargeWindows,
    process.DoHLTElectronSiStrip,
    process.DoHLTTau, 
    process.DoHLTBTag,
    process.DoHLTAlCaECALPhiSym,
    process.DoHLTAlCaPi0Eta1E31,
    process.DoHLTIsoTrack,
    process.DoHLTMinBiasPixelTracks,
    process.analyzeThis )
