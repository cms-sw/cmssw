import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIM")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.8 $'),
    name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/DPGAnalysis/Skims/python/MuPDSkim_cfg.py,v $'),
    annotation = cms.untracked.string('Mu skim')
)

#
#
# This is for testing purposes.
#
#
##run143960
process.source = cms.Source("PoolSource",
                           fileNames = cms.untracked.vstring('/store/data/Run2010A/Mu/RECO/v4/000/144/114/9C954151-32B4-DF11-BB88-001D09F27003.root',
                                                             '/store/data/Run2010A/Mu/RECO/v4/000/144/114/5C9CA515-20B4-DF11-9D62-0030487A3DE0.root',
                                                             '/store/data/Run2010A/Mu/RECO/v4/000/144/114/00CA69A9-1EB4-DF11-B869-0030487CD6D2.root'),
                           secondaryFileNames = cms.untracked.vstring('/store/data/Run2010A/Mu/RAW/v1/000/144/114/70E434AF-04B4-DF11-B178-0030487CD718.root',
                                                                      '/store/data/Run2010A/Mu/RAW/v1/000/144/114/C25ACF12-F3B3-DF11-8BB6-0030487CD700.root',
                                                                      '/store/data/Run2010A/Mu/RAW/v1/000/144/114/643C84A4-F1B3-DF11-8809-003048F0258C.root')
   )

process.source.inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
    )


#------------------------------------------
# Load standard sequences.
#------------------------------------------
process.load('Configuration/StandardSequences/MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration/StandardSequences/GeometryIdeal_cff')


process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'GR_R_38X_V13::All'

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

#################################WZFilter############################################

process.load("DPGAnalysis.Skims.WZMuSkim_cff")

process.ZMuSkimPath      = cms.Path(process.diMuonSelSeq)
process.WtcMetSkimPath   = cms.Path(process.tcMetWMuNuSeq)
process.WpfMetSkimPath   = cms.Path(process.pfMetWMuNuSeq)


process.SKIMStreamWZMu = cms.OutputModule("PoolOutputModule",
                                          # splitLevel = cms.untracked.int32(0),
                                          outputCommands = process.FEVTEventContent.outputCommands,
                                          fileName = cms.untracked.string('/tmp/azzi/WZMuStream.root'),
                                          dataset = cms.untracked.PSet(dataTier = cms.untracked.string('RAW-RECO'),
                                                                       filterName = cms.untracked.string('WZMuFilter')),
                                          SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('ZMuSkimPath',
                                                                                                       'WtcMetSkimPath',
                                                                                                       'WpfMetSkimPath')
                                                                            ))


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
                                     dataset = cms.untracked.PSet(dataTier = cms.untracked.string('RAW-RECO'),
                                                                  filterName = cms.untracked.string('Skim_logerror')),
                                     SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring("logerrorpath")
                                                                       ))

#======================
######################################TPG Performance SKIMS#####################################

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

process.outpath = cms.EndPath(process.outlogerr+process.SKIMStreamWZMu+process.outTPGSkim)
