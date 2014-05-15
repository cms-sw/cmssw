import FWCore.ParameterSet.Config as cms

##################################################################

# useful options
gtDigisExist=0  # =1 use existing gtDigis on the input file, =0 extract gtDigis from the RAW data collection
isData=0 # =1 running on real data, =1 running on MC

OUTPUT_HIST='hltbits.root'
NEVTS=-1

##################################################################

process = cms.Process("ANALYSIS")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'/store/data/Run2012C/MinimumBias/RAW-RECO/HLTPhysics-Tier1PromptSkim-v3/000/198/969/00000/C8575CDA-A8D1-E111-A17B-E0CB4E1A1193.root'
#'/store/data/Run2012C/LP_Jets1/RAW/v1/000/198/449/F0CDFC81-35C8-E111-8D6C-003048678110.root'
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
#process.GlobalTag.globaltag = 'MC_31X_V9::All'
#process.GlobalTag.globaltag = 'START42_V9::All'
process.GlobalTag.globaltag = 'GR_R_52_V1::All'
process.GlobalTag.pfnPrefix=cms.untracked.string('frontier://FrontierProd/')

process.load('Configuration/StandardSequences/SimL1Emulator_cff')

# OpenHLT specificss
# Define the HLT reco paths
process.load("HLTrigger.HLTanalyzers.HLT_FULL_cff")
# Remove the PrescaleService which, in 31X, it is expected once HLT_XXX_cff is imported

process.DQM = cms.Service( "DQM",)
process.DQMStore = cms.Service( "DQMStore",)

# AlCa OpenHLT specific settings

# Define the analyzer modules
process.load("HLTrigger.HLTanalyzers.HLTBitAnalyser_cfi")
process.hltbitanalysis.hltresults = cms.InputTag( 'TriggerResults','','TEST' )
process.hltbitanalysis.RunParameters.HistogramFile=OUTPUT_HIST

if (gtDigisExist):
    process.analyzeThis = cms.Path( process.hltbitanalysis )
else:
    process.analyzeThis = cms.Path(process.HLTBeginSequence + process.hltbitanalysis )
    process.hltbitanalysis.l1GtReadoutRecord = cms.InputTag( 'hltGtDigis','',process.name_() )


# pdt
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

# Schedule the whole thing
process.schedule = cms.Schedule( 
    process.analyzeThis )

#########################################################################################
#
#nc=0
if (isData):  # replace all instances of "rawDataCollector" with "source" in InputTags
    from FWCore.ParameterSet import Mixins
    for module in process.__dict__.itervalues():
        if isinstance(module, Mixins._Parameterizable):
            for parameter in module.__dict__.itervalues():
                if isinstance(parameter, cms.InputTag):
                    if parameter.moduleLabel == 'rawDataCollector':
                        parameter.moduleLabel = 'source'
                        #print "Replacing in module: ", module
                        #nc=nc+1
#print "Number of replacements: ", nc
