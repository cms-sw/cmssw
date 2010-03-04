import FWCore.ParameterSet.Config as cms

process = cms.Process("EWKDiElectronSkim")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# source
process.source = cms.Source("PoolSource", 
     fileNames = cms.untracked.vstring(
    'rfio:/tmp/ikesisog/10A96AFC-E17C-DE11-A90E-001D0967D9CC.root'
    )
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.GlobalTag.globaltag = cms.string('MC_31X_V3::All')


# HLT filter
import HLTrigger.HLTfilters.hltHighLevel_cfi
process.EWK_DiElectronHLTFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()

# Uncomment this to access 8E29 menu and filter on it
process.EWK_DiElectronHLTFilter.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT8E29")
process.EWK_DiElectronHLTFilter.HLTPaths = ["HLT_Ele15_LW_L1R"]


#   Make a collection of good SuperClusters.
#
#   Before selection is made, merge the Barrel and EndCap SC's.
process.superClusterMerger =  cms.EDProducer("EgammaSuperClusterMerger",
                                    src = cms.VInputTag(cms.InputTag('correctedHybridSuperClusters'), cms.InputTag('correctedMulti5x5SuperClustersWithPreshower'))
                                  )

#   Get the above merged SC's and select the particle (gamma) to greate SC's Candidates.
process.superClusterCands = cms.EDProducer("ConcreteEcalCandidateProducer",
                                    src = cms.InputTag("superClusterMerger"), particleType = cms.string('gamma')
                                  )

#   Get the above SC's Candidates and place a cut on their Et.
process.goodSuperClusters = cms.EDFilter("CandViewRefSelector",
                                    src = cms.InputTag("superClusterCands"),
                                    cut = cms.string('et > 20.0'),
                                    filter = cms.bool(True)
                                )

process.superClusterFilter = cms.Sequence(process.superClusterMerger + process.superClusterCands + process.goodSuperClusters)


#   Make a collections on good Electrons.
#
process.goodElectrons = cms.EDFilter("CandViewSelector",
                                src = cms.InputTag("gsfElectrons"),
                                cut = cms.string('pt > 20.0'),
                                filter = cms.bool(True)                                
)

#   Filter the above two collections (good SuperClusters and good Electrons)
#
process.electronSuperClusterCombiner = cms.EDFilter("CandViewShallowCloneCombiner",
                                            filter = cms.bool(True),
                                            checkCharge = cms.bool(False),
                                            cut = cms.string('mass > 3.0'),
                                            decay = cms.string('goodElectrons goodSuperClusters')
                                            )

process.electronSuperClusterCounter = cms.EDFilter("CandViewCountFilter",
                                            src = cms.InputTag("electronSuperClusterCombiner"),
                                            minNumber = cms.uint32(1)
                                          )

process.electronSuperClusterFilter = cms.Sequence(process.electronSuperClusterCombiner + process.electronSuperClusterCounter)


# Skim path
process.EWK_DiElectronSkimPath = cms.Path(process.EWK_DiElectronHLTFilter +
                                          process.goodElectrons + 
                                          process.superClusterFilter + 
                                          process.electronSuperClusterFilter
                                         )


# Output module configuration
from Configuration.EventContent.EventContent_cff import *

EWK_DiElectronSkimEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
EWK_DiElectronSkimEventContent.outputCommands.extend(AODEventContent.outputCommands)

EWK_DiElectronSkimEventContent.outputCommands.extend(
      cms.untracked.vstring('drop *',
            "keep recoSuperClusters_*_*_*",
            "keep *_gsfElectrons_*_*",
            "keep recoGsfTracks_electronGsfTracks_*_*",
            "keep *_gsfElectronCores_*_*",
            "keep *_correctedHybridSuperClusters_*_*",
            "keep *_correctedMulti5x5SuperClustersWithPreshower_*_*",
            "keep edmTriggerResults_*_*_*",
            "keep recoCaloMETs_*_*_*",
            "keep recoMETs_*_*_*",
            "keep *_particleFlow_electrons_*",
            "keep *_pfMet_*_*",
            "keep *_multi5x5SuperClusterWithPreshower_*_*",
            "keep recoVertexs_*_*_*",
            "keep *_hltTriggerSummaryAOD_*_*",
            "keep floatedmValueMap_*_*_*",
            "keep recoBeamSpot_*_*_*" )
)

EWK_DiElectronSkimEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring(
           'EWK_DiElectronSkimPath')
    )
)

process.EWK_DiElectronSkimOutputModule = cms.OutputModule("PoolOutputModule",
                                            EWK_DiElectronSkimEventContent,
                                            EWK_DiElectronSkimEventSelection,
                                            dataset = cms.untracked.PSet(
                                                filterName = cms.untracked.string('EWKSKIMEMET'),
                                                dataTier = cms.untracked.string('USER')
                                            ),
                                            fileName = cms.untracked.string('EWKDiElectronSkim.root')
)

process.outpath = cms.EndPath(process.EWK_DiElectronSkimOutputModule)


