import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIM")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.5 $'),
    name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/DPGAnalysis/Skims/python/ZeroBiasPDSkim_cfg.py,v $'),
    annotation = cms.untracked.string('Combined ZeroBias skim')
)
# selection eff on 1000 events
# file:/tmp/malgeri/ZB_vertex.root
# /tmp/malgeri/ZB_vertex.root ( 45 events, 15799040 bytes )

#
#
# This is for testing purposes.
#
#
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
# run 136066 lumi~500
'/store/data/Run2010A/ZeroBias/RECO/v1/000/136/066/DE81B1E0-4866-DF11-A76D-0030487CD906.root'),
                           secondaryFileNames = cms.untracked.vstring(
'/store/data/Run2010A/ZeroBias/RAW/v1/000/136/066/FE1DCAEF-3C66-DF11-A903-000423D98E30.root')
)

process.source.inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)


#------------------------------------------
# Load standard sequences.
#------------------------------------------
process.load('Configuration/StandardSequences/MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration/StandardSequences/GeometryIdeal_cff')


process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'GR10_P_V6::All' 

process.load("Configuration/StandardSequences/RawToDigi_Data_cff")
process.load("Configuration/StandardSequences/Reconstruction_cff")
process.load('Configuration/EventContent/EventContent_cff')

process.FEVTEventContent.outputCommands.append('drop *_MEtoEDMConverter_*_*')

#
#  Load common sequences
#
#process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskAlgoTrigConfig_cff')
#process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
#process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')

##################################good vertex############################################


process.primaryVertexFilter = cms.EDFilter("VertexSelector",
   src = cms.InputTag("offlinePrimaryVertices"),
   cut = cms.string("!isFake && ndof > 4 && abs(z) <= 15 && position.Rho <= 2"), # tracksSize() > 3 for the older cut
   filter = cms.bool(True),   # otherwise it won't filter the events, just produce an empty vertex collection.
)


process.noscraping = cms.EDFilter("FilterOutScraping",
applyfilter = cms.untracked.bool(True),
debugOn = cms.untracked.bool(False),
numtrack = cms.untracked.uint32(10),
thresh = cms.untracked.double(0.25)
)

process.goodvertex=cms.Path(process.primaryVertexFilter+process.noscraping)


process.gvout = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('/tmp/azzi/ZB_vertex.root'),
    outputCommands = process.FEVTEventContent.outputCommands,
    dataset = cms.untracked.PSet(
    	      dataTier = cms.untracked.string('RAW-RECO'),
    	      filterName = cms.untracked.string('GOODVERTEX')),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('goodvertex')
    )
)


#################################logerrorharvester############################################
process.load("FWCore.Modules.logErrorFilter_cfi")
from Configuration.StandardSequences.RawToDigi_Data_cff import gtEvmDigis

process.gtEvmDigis = gtEvmDigis.clone()
process.stableBeam = cms.EDFilter("HLTBeamModeFilter",
                                  L1GtEvmReadoutRecordTag = cms.InputTag("gtEvmDigis"),
                                  AllowedBeamMode = cms.vuint32(11),
                                  saveTags = cms.bool(False)
                                  )

process.logerrorpath=cms.Path(process.gtEvmDigis+process.stableBeam+process.logErrorFilter)

process.outlogerr = cms.OutputModule("PoolOutputModule",
                               outputCommands =  process.FEVTEventContent.outputCommands,
                               fileName = cms.untracked.string('/tmp/azzi/logerror_filter.root'),
                               dataset = cms.untracked.PSet(
                                  dataTier = cms.untracked.string('RAW-RECO'),
                                  filterName = cms.untracked.string('Skim_logerror')),
                               
                               SelectEvents = cms.untracked.PSet(
    SelectEvents = cms.vstring("logerrorpath")
    ))

#===========================================================

###########################################################################
process.options = cms.untracked.PSet(
 wantSummary = cms.untracked.bool(True)
)

#Killed gvskim 
#process.outpath = cms.EndPath(process.gvout+process.outlogerr)
process.outpath = cms.EndPath(process.outlogerr)



 
