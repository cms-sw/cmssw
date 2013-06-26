import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIM")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.3 $'),
    name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/DPGAnalysis/Skims/python/MinBiasPDSkim_ONLYHSCP_cfg.py,v $'),
    annotation = cms.untracked.string('Combined MinBias skim')
)
# selection eff. on 1000 events
#file:/tmp/azzi/Background.root
#/tmp/azzi/Background.root ( 0 events, 664191 bytes )
#file:/tmp/azzi/MinBiascscskimEvents.root
#/tmp/azzi/MinBiascscskimEvents.root ( 0 events, 664212 bytes )
#file:/tmp/azzi/MuonDPGSkim.root
#/tmp/azzi/MuonDPGSkim.root ( 95 events, 40887080 bytes )
#file:/tmp/azzi/StoppedHSCP_filter.root
#/tmp/azzi/StoppedHSCP_filter.root ( 124 events, 28507733 bytes )
#file:/tmp/azzi/ValSkim.root
#/tmp/azzi/ValSkim.root ( 22 events, 9588726 bytes )
#file:/tmp/azzi/ecalrechitfilter.root
#/tmp/azzi/ecalrechitfilter.root ( 23 events, 14583758 bytes )
#file:/tmp/azzi/logerror_filter.root
#/tmp/azzi/logerror_filter.root ( 14 events, 9389296 bytes )
#file:/tmp/azzi/TPGSkim.root
#/tmp/azzi/TPGSkim.root ( 46 events, 20271643 bytes )

#
#
# This is for testing purposes.
#
#
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
# run 136066 lumi~500
'/store/data/Run2010A/MinimumBias/RECO/v1/000/136/066/18F6DB82-5566-DF11-B289-0030487CAF0E.root'),
                           secondaryFileNames = cms.untracked.vstring(
'/store/data/Run2010A/MinimumBias/RAW/v1/000/136/066/38D48BED-3C66-DF11-88A5-001D09F27003.root')
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
process.GlobalTag.globaltag = 'GR_R_36X_V12B::All' 

process.load("Configuration/StandardSequences/RawToDigi_Data_cff")
process.load("Configuration/StandardSequences/Reconstruction_cff")
process.load('Configuration/EventContent/EventContent_cff')

#drop collections created on the fly
process.FEVTEventContent.outputCommands.append("drop *_MEtoEDMConverter_*_*")
process.FEVTEventContent.outputCommands.append("drop *_*_*_SKIM")

#
#  Load common sequences
#
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskAlgoTrigConfig_cff')
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')

##################################stoppedHSCP############################################


# this is for filtering on HLT path
process.hltstoppedhscp = cms.EDFilter("HLTHighLevel",
     TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
     HLTPaths = cms.vstring("HLT_StoppedHSCP*"), # provide list of HLT paths (or patterns) you want
     eventSetupPathsKey = cms.string(''), # not empty => use read paths from AlCaRecoTriggerBitsRcd via this key
     andOr = cms.bool(True),             # how to deal with multiple triggers: True (OR) accept if ANY is true, False (AND) accept if ALL are true
     throw = cms.bool(False),    # throw exception on unknown path names
     saveTags = cms.bool(False)
)

process.HSCP=cms.Path(process.hltstoppedhscp)

process.outHSCP = cms.OutputModule("PoolOutputModule",
                               outputCommands =  process.FEVTEventContent.outputCommands,
                               fileName = cms.untracked.string('/tmp/malgeri/StoppedHSCP_filter.root'),
                               dataset = cms.untracked.PSet(
                                  dataTier = cms.untracked.string('RAW-RECO'),
                                  filterName = cms.untracked.string('Skim_StoppedHSCP')),
                               
                               SelectEvents = cms.untracked.PSet(
    SelectEvents = cms.vstring("HSCP")
    ))



process.options = cms.untracked.PSet(
 wantSummary = cms.untracked.bool(True)
)

process.outpath = cms.EndPath(process.outHSCP)



 
