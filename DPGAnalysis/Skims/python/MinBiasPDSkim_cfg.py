import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIM")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.37 $'),
    name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/DPGAnalysis/Skims/python/MinBiasPDSkim_cfg.py,v $'),
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

###########################################################################################
#------------------------------------------
# parameters for the CSCSkim module
#------------------------------------------
process.load("DPGAnalysis/Skims/CSCSkim_cfi")


#set to minimum activity
process.cscSkim.minimumSegments = 1
process.cscSkim.minimumHitChambers = 1

# this is for filtering on HLT path
process.hltBeamHalo = cms.EDFilter("HLTHighLevel",
     TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
     HLTPaths = cms.vstring('HLT_CSCBeamHalo','HLT_CSCBeamHaloOverlapRing1','HLT_CSCBeamHaloOverlapRing','HLT_CSCBeamHaloRing2or3'), # provide list of HLT paths (or patterns) you want
     eventSetupPathsKey = cms.string(''), # not empty => use read paths from AlCaRecoTriggerBitsRcd via this key
     andOr = cms.bool(True),             # how to deal with multiple triggers: True (OR) accept if ANY is true, False (AND) accept if ALL are true
     throw = cms.bool(False),    # throw exception on unknown path names
     saveTags = cms.bool(False)
 )

#### the path
process.cscHaloSkim = cms.Path(process.hltBeamHalo+process.cscSkim)



#### output 
process.outputBeamHaloSkim = cms.OutputModule("PoolOutputModule",
    outputCommands = process.FEVTEventContent.outputCommands,
    fileName = cms.untracked.string("/tmp/azzi/MinBiascscskimEvents.root"),
    dataset = cms.untracked.PSet(
      dataTier = cms.untracked.string('RAW-RECO'),
      filterName = cms.untracked.string('CSCSkim_BeamHalo_MinBias')
    ),
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('cscHaloSkim'))
)


########################## Muon tracks Filter ############################
process.muonSkim=cms.EDFilter("CandViewCountFilter", 
                 src =cms.InputTag("muons"), minNumber = cms.uint32(1))
process.muonTracksSkim = cms.Path(process.muonSkim)


###########################################################################

process.outputMuonSkim = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('/tmp/azzi/MuonSkim.root'),
    outputCommands = cms.untracked.vstring('keep *','drop *_MEtoEDMConverter_*_*'),
    dataset = cms.untracked.PSet(
    	      dataTier = cms.untracked.string('RECO'),
    	      filterName = cms.untracked.string('Muon_skim')),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring("muonTracksSkim")
    )
)
####################################################################################

##################################stoppedHSCP############################################


# this is for filtering on HLT path
process.hltstoppedhscp = cms.EDFilter("HLTHighLevel",
     TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
     HLTPaths = cms.vstring("HLT_StoppedHSCP*"), # provide list of HLT paths (or patterns) you want
     eventSetupPathsKey = cms.string(''), # not empty => use read paths from AlCaRecoTriggerBitsRcd via this key
     andOr = cms.bool(True),             # how to deal with multiple triggers: True (OR) accept if ANY is true, False (AND) accept if ALL are true
     throw = cms.bool(False)    # throw exception on unknown path names
 )

process.HSCP=cms.Path(process.hltstoppedhscp)

process.outHSCP = cms.OutputModule("PoolOutputModule",
                               outputCommands =  process.FEVTEventContent.outputCommands,
                               fileName = cms.untracked.string('/tmp/azzi/StoppedHSCP_filter.root'),
                               dataset = cms.untracked.PSet(
                                  dataTier = cms.untracked.string('RAW-RECO'),
                                  filterName = cms.untracked.string('Skim_StoppedHSCP')),
                               
                               SelectEvents = cms.untracked.PSet(
    SelectEvents = cms.vstring("HSCP")
    ))

###########################################################################################
#------------------------------------------
# parameters for the PFGCollisions skim3
#------------------------------------------
process.load('HLTrigger.special.hltPhysicsDeclared_cfi')
process.hltPhysicsDeclared.L1GtReadoutRecordTag = 'gtDigis'

process.hltbeamgas = cms.EDFilter("HLTHighLevel",
TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
  HLTPaths = cms.vstring('HLT_L1_BptxXOR_BscMinBiasOR'), # provide list of HLT paths (or patterns) you want
  eventSetupPathsKey = cms.string(''),
  andOr              = cms.bool(True),
  throw              = cms.bool(False),
  saveTags           = cms.bool(False)

)

#### the path
process.pfgskim3noncross = cms.Path(process.hltPhysicsDeclared*process.hltbeamgas)

#### output
process.outputpfgskim3 = cms.OutputModule("PoolOutputModule",
 outputCommands = process.FEVTEventContent.outputCommands,
 fileName = cms.untracked.string("/tmp/azzi/Background.root"),
 dataset = cms.untracked.PSet(
   dataTier = cms.untracked.string('RAW-RECO'),
   filterName = cms.untracked.string('BEAMBKGV3')
 ),
 SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('pfgskim3noncross'))
)

###########################################################################################
###########################################################################################

#===========================================================

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
###########################ngood event per lumi##########################################
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

###Tracks selection
process.trackSelector  =cms.EDFilter("TrackSelector",
                                    src = cms.InputTag("generalTracks"),
                                     cut = cms.string('quality("highPurity")')     
                                     )

#process.trackSelector = cms.EDProducer("QualityFilter",
#                                       TrackQuality = cms.string('highPurity'),
#                                       recTracks = cms.InputTag("generalTracks")
#                                       )

process.trackFilter = cms.EDFilter("TrackCountFilter",
                                   src = cms.InputTag("trackSelector"),
                                   minNumber = cms.uint32(10)
                                   )

process.nottoomanytracks = cms.EDFilter("NMaxPerLumi",
                                        nMaxPerLumi = cms.uint32(8)
                                        )
process.relvaltrackskim = cms.Path(process.primaryVertexFilter+process.noscraping+
                                   process.trackSelector + process.trackFilter + process.nottoomanytracks )

### muon selection
process.muonSelector = cms.EDFilter("MuonSelector",
                                    src = cms.InputTag("muons"),
                                    cut = cms.string(" isGlobalMuon && isTrackerMuon && pt > 3")
                                    )
process.muonFilter = cms.EDFilter("MuonCountFilter",
                                  src = cms.InputTag("muonSelector"),
                                  minNumber = cms.uint32(1)
                                  )
process.nottoomanymuons = cms.EDFilter("NMaxPerLumi",
                                       nMaxPerLumi = cms.uint32(2)
                                       )
process.relvalmuonskim = cms.Path(process.primaryVertexFilter+process.noscraping+
                                  process.muonSelector + process.muonFilter + process.nottoomanymuons )

#### output 
process.outputvalskim = cms.OutputModule("PoolOutputModule",
                                         outputCommands = process.FEVTEventContent.outputCommands,
                                         fileName = cms.untracked.string("/tmp/azzi/ValSkim.root"),
                                         dataset = cms.untracked.PSet(
    dataTier = cms.untracked.string('RAW-RECO'),
    filterName = cms.untracked.string('valskim')
    ),
                                         SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('relvaltrackskim','relvalmuonskim')
                                                                           ))


###########################################################################
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

#process.outpath = cms.EndPath(process.outputBeamHaloSkim+process.outputMuonSkim+process.outHSCP+process.outputpfgskim3+process.outlogerr+process.outputvalskim+process.outTPGSkim)
#BeamHalo removed
process.outpath = cms.EndPath(process.outputMuonSkim+process.outHSCP+process.outputpfgskim3+process.outlogerr+process.outputvalskim+process.outTPGSkim)


 
