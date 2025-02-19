import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIM")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    name = cms.untracked.string('$Source' ),
    annotation = cms.untracked.string('TPG skim')
)

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

process.source.inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*", "drop L1GlobalTriggerObjectMapRecord_hltL1GtObjectMap__HLT")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(500)
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

#drop collections created on the fly
process.FEVTEventContent.outputCommands.append("drop *_MEtoEDMConverter_*_*")
process.FEVTEventContent.outputCommands.append("drop *_*_*_SKIM")

#
#  Load common sequences
#
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskAlgoTrigConfig_cff')
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')

##################################TPG Skims############################################


process.load('DPGAnalysis/Skims/singleMuonSkim_cff')
process.load('DPGAnalysis/Skims/singleElectronSkim_cff')
process.load('DPGAnalysis/Skims/muonTagProbeFilters_cff')
process.load('DPGAnalysis/Skims/electronTagProbeFilters_cff')
process.load('DPGAnalysis/Skims/singlePhotonSkim_cff')
process.load('DPGAnalysis/Skims/jetSkim_cff')
process.load('DPGAnalysis/Skims/METSkim_cff')
process.load('DPGAnalysis/Skims/singlePfTauSkim_cff')

#process.singleMuPt20SkimPath=cms.Path(process.singleMuPt20RecoQualitySeq)
#process.singleMuPt15SkimPath=cms.Path(process.singleMuPt15RecoQualitySeq)
#process.singleMuPt10SkimPath=cms.Path(process.singleMuPt10RecoQualitySeq)
process.singleMuPt5SkimPath=cms.Path(process.singleMuPt5RecoQualitySeq)
#process.singleElectronPt20SkimPath=cms.Path(process.singleElectronPt20RecoQualitySeq)
#process.singleElectronPt15SkimPath=cms.Path(process.singleElectronPt15RecoQualitySeq)
#process.singleElectronPt10SkimPath=cms.Path(process.singleElectronPt10RecoQualitySeq)
process.singleElectronPt5SkimPath=cms.Path(process.singleElectronPt5RecoQualitySeq)
#process.singlePhotonPt20SkimPath=cms.Path(process.singlePhotonPt20QualitySeq)
#process.singlePhotonPt15SkimPath=cms.Path(process.singlePhotonPt15QualitySeq)
#process.singlePhotonPt10SkimPath=cms.Path(process.singlePhotonPt10QualitySeq)
process.singlePhotonPt5SkimPath=cms.Path(process.singlePhotonPt5QualitySeq)
#process.muonZMMSkimPath=cms.Path(process.muonZMMRecoQualitySeq)
process.muonJPsiMMSkimPath=cms.Path(process.muonJPsiMMRecoQualitySeq)
#process.electronZEESkimPath=cms.Path(process.electronZEERecoQualitySeq)
process.jetSkimPath=cms.Path(process.jetRecoQualitySeq)
#process.METSkimPath=cms.Path(process.METQualitySeq)
process.singlePfTauPt15SkimPath=cms.Path(process.singlePfTauPt15QualitySeq) 

process.outTPGSkim = cms.OutputModule("PoolOutputModule",
    outputCommands = process.FEVTHLTALLEventContent.outputCommands,
    fileName = cms.untracked.string("/tmp/azzi/TPGSkim.root"),
    dataset = cms.untracked.PSet(
      dataTier = cms.untracked.string('USER'),
      filterName = cms.untracked.string('TPGSkim')
    ),
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring(
                                                                 #'singleMuPt20SkimPath',
                                                                 #'singleMuPt15SkimPath',
                                                                 #'singleMuPt10SkimPath',
                                                                 'singleMuPt5SkimPath',
                                                                 #'singleElectronPt20SkimPath',
                                                                 #'singleElectronPt15SkimPath',
                                                                 #'singleElectronPt10SkimPath',
                                                                 'singleElectronPt5SkimPath',
                                                                 #'singlePhotonPt20SkimPath',
                                                                 #'singlePhotonPt15SkimPath',
                                                                 #'singlePhotonPt10SkimPath',
                                                                 'singlePhotonPt5SkimPath',
                                                                 #'muonZMMSkimPath',
                                                                 'muonJPsiMMSkimPath',
                                                                 #'electronZEESkimPath',
                                                                 'jetSkimPath',
                                                                 #'METSkimPath',
                                                                 'singlePfTauPt15SkimPath'))
)


###########################################################################


process.options = cms.untracked.PSet(
 wantSummary = cms.untracked.bool(True)
)

process.outpath = cms.EndPath(process.outTPGSkim)

