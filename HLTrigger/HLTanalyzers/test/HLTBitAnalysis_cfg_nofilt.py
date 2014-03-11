import FWCore.ParameterSet.Config as cms

##################################################################

# useful options
gtDigisExist=0  # =1 use existing gtDigis on the input file, =0 extract gtDigis from the RAW data collection
isData=0 # =1 running on real data, =0 running on MC

OUTPUT_HIST='hltbitsmc.root'
NEVTS=-1

##################################################################

process = cms.Process("ANALYSIS")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.source = cms.Source("PoolSource",
    skipBadFiles = cms.untracked.bool( True ),
    fileNames = cms.untracked.vstring(
'root://xrootd.unl.edu//store/mc/Summer13dr53X/QCD_Pt_30_80_EMEnriched_TuneZ2star_13TeV-pythia6/GEN-SIM-RAW/PU25bx25_START53_V19D-v1/20000/0042293B-89E1-E211-A506-485B39800B97.root'
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( NEVTS ),
    skipBadFiles = cms.bool(True)
    )

process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'START53_V19D::All'
process.GlobalTag.pfnPrefix=cms.untracked.string('frontier://FrontierProd/')

process.load('Configuration/StandardSequences/SimL1Emulator_cff')
process.load("HLTrigger.HLTanalyzers.HLTopen_cff")

# OpenHLT specificss
# Define the HLT reco paths
#process.load("HLTrigger.HLTanalyzers.HLT_FULL_cff")
# Remove the PrescaleService which, in 31X, it is expected once HLT_XXX_cff is imported

process.DQM = cms.Service( "DQM",)
process.DQMStore = cms.Service( "DQMStore",)

# AlCa OpenHLT specific settings

# Define the analyzer modules
process.load("HLTrigger.HLTanalyzers.HLTBitAnalyser_cfi")
process.hltbitanalysis.hltresults = cms.InputTag( 'TriggerResults','','HLT' )
process.hltbitanalysis.RunParameters.HistogramFile=OUTPUT_HIST
process.hltbitanalysis.l1GtObjectMapRecord             = cms.InputTag("l1L1GtObjectMap::RECO")
process.hltbitanalysis.l1GtReadoutRecord               = cms.InputTag("gtDigis::RECO")
process.hltbitanalysis.l1extramc                       = cms.string('l1extraParticles')
process.hltbitanalysis.l1extramu                       = cms.string('l1extraParticles')


if (gtDigisExist):
    process.analyzeA = cms.Path(process.hltbitanalysis)
else:
    process.analyzeA = cms.Path(process.HLTBeginSequence * process.hltbitanalysis)
    process.hltbitanalysis.l1GtReadoutRecord = cms.InputTag( 'hltGtDigis','',process.name_() )
    
# pdt
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

# Schedule the whole thing
process.schedule = cms.Schedule( 
    process.analyzeA)

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
