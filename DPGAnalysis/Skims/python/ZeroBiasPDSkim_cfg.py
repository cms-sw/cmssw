import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIM")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.23 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/DPGAnalysis/Skims/python/MinBiasPDSkim_cfg.py,v $'),
    annotation = cms.untracked.string('Combined ZeroBias skim')
)

#
#
# This is for testing purposes.
#
#
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
# run 133874
'/store/data/Commissioning10/ZeroBias/RECO/v9/000/133/874/A4E09DD6-7B4F-DF11-95EA-001D09F29146.root',
'/store/data/Commissioning10/ZeroBias/RECO/v9/000/133/874/62DA9F76-814F-DF11-9ABC-001D09F25456.root',
'/store/data/Commissioning10/ZeroBias/RECO/v9/000/133/874/5AFCCDAB-9F4F-DF11-9CA2-001D09F24F1F.root',
'/store/data/Commissioning10/ZeroBias/RECO/v9/000/133/874/021C6706-804F-DF11-AF31-001D09F27067.root'),
                           secondaryFileNames = cms.untracked.vstring(
'/store/data/Commissioning10/ZeroBias/RAW/v4/000/133/874/F0BAD5CA-794F-DF11-9575-0019B9F730D2.root',
'/store/data/Commissioning10/ZeroBias/RAW/v4/000/133/874/7CEF7709-744F-DF11-ABCA-001D09F29321.root',
'/store/data/Commissioning10/ZeroBias/RAW/v4/000/133/874/508AC045-714F-DF11-AFD9-000423D8FA38.root',
'/store/data/Commissioning10/ZeroBias/RAW/v4/000/133/874/5056F04D-8B4F-DF11-9018-001D09F2545B.root',
'/store/data/Commissioning10/ZeroBias/RAW/v4/000/133/874/0878ECFD-784F-DF11-8D20-0030487CD7EA.root')
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
process.GlobalTag.globaltag = 'GR10_P_V5::All' 

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
    fileName = cms.untracked.string('/tmp/malgeri/ZB_vertex.root'),
    outputCommands = process.FEVTEventContent.outputCommands,
    dataset = cms.untracked.PSet(
    	      dataTier = cms.untracked.string('RAW-RECO'),
    	      filterName = cms.untracked.string('GOODVERTEX')),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('goodvertex')
    )
)

###########################################################################
process.options = cms.untracked.PSet(
 wantSummary = cms.untracked.bool(True)
)

process.outpath = cms.EndPath(process.gvout)



 
