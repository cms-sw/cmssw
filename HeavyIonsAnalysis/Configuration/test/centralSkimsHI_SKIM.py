# Auto generated configuration file
# using: 
# Revision: 1.232.2.6 
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: centralSkimsHI -s SKIM:DiJet+Photon+ZMM+ZEE --conditions GR10_P_V12::All --scenario HeavyIons --filein=/store/hidata/HIRun2010/HIAllPhysics/RECO/PromptReco-v1/000/150/063/B497BEDB-8BE8-DF11-B09D-0030487A18F2.root --data --no_exec
import FWCore.ParameterSet.Config as cms

process = cms.Process('SKIM')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.SkimsHeavyIons_cff')
process.load('Configuration.StandardSequences.ReconstructionHeavyIons_cff')
process.load('RecoHI.HiEgammaAlgos.HiElectronSequence_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.EventContent.EventContentHeavyIons_cff')

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.6 $'),
    annotation = cms.untracked.string('centralSkimsHI nevts:1'),
    name = cms.untracked.string('PyReleaseValidation')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/hidata/HIRun2010/HICorePhysics/RECO/PromptReco-v3/000/151/353/DC04F2F3-32F2-DF11-BD72-0030487C5CE2.root'),
    secondaryFileNames = cms.untracked.vstring(
    '/store/hidata/HIRun2010/HICorePhysics/RAW/v1/000/151/353/00B95950-00F2-DF11-9485-001D09F252DA.root')
)

process.options = cms.untracked.PSet(
    #wantSummary = cms.untracked.bool(True)
)

#process.Timing = cms.Service("Timing")

# Output definition

# Additional output definition
process.SKIMStreamDiJet = cms.OutputModule("PoolOutputModule",
    process.FEVTEventContent,                                           
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('diJetSkimPath')
    ),
    fileName = cms.untracked.string('DiJet.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('DiJet'),
        dataTier = cms.untracked.string('RAW-RECO')
    )
)
process.SKIMStreamPhoton = cms.OutputModule("PoolOutputModule",
    process.FEVTEventContent,
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('photonSkimPath')
    ),
    fileName = cms.untracked.string('Photon.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('Photon'),
        dataTier = cms.untracked.string('RAW-RECO')
    )
)
process.SKIMStreamZEE = cms.OutputModule("PoolOutputModule",
    process.FEVTEventContent,
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('zEESkimPath')
    ),
    fileName = cms.untracked.string('ZEE.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('ZEE'),
        dataTier = cms.untracked.string('RAW-RECO')
    )
)
process.SKIMStreamZMM = cms.OutputModule("PoolOutputModule",
    process.FEVTEventContent,
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('zMMSkimPath')
    ),
    fileName = cms.untracked.string('ZMM.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('ZMM'),
        dataTier = cms.untracked.string('RAW-RECO')
    )
)

# Other statements
process.GlobalTag.globaltag = 'GR10_P_V12::All'

# Valid vertex filter
process.primaryVertexFilter = cms.EDFilter("VertexSelector",
    src = cms.InputTag("hiSelectedVertex"),
    cut = cms.string("!isFake && abs(z) <= 25 && position.Rho <= 2"), 
    filter = cms.bool(True),   # otherwise it won't filter the events
    )

# Refine photon cuts
process.goodPhotons.cut = cms.string('et > 20 && hadronicOverEm < 0.1 && r9 > 0.8 && sigmaIetaIeta > 0.002')

# Looser photon cuts for ZEE
process.goodPhotonsForZEE = process.goodPhotons.clone(
    cut=cms.string('et > 20 && hadronicOverEm < 0.2 && r9 > 0.5 && sigmaIetaIeta > 0.002')
    )
process.goodCleanPhotonsForZEE = process.goodPhotonsForZEE.clone(src=cms.InputTag("cleanPhotons"))
process.twoPhotonFilter.src = cms.InputTag("goodPhotonsForZEE")
process.photonCombiner.decay = cms.string('goodCleanPhotonsForZEE goodCleanPhotonsForZEE')
process.fullZEESkimSequence = cms.Sequence(process.hltPhotonHI
                                           * process.primaryVertexFilter
                                           * process.goodPhotonsForZEE
                                           * process.twoPhotonFilter
                                           * process.hiPhotonCleaningSequence
                                           * process.goodCleanPhotonsForZEE
                                           * process.photonCombiner * process.photonPairCounter
                                           * process.siPixelRecHits * process.siStripMatchedRecHits
                                           * process.hiPrimSeeds * process.hiElectronSequence)


# Higher trigger thresholds
process.hltJetHI.HLTPaths = ["HLT_HIJet50U_Core"]
process.hltPhotonHI.HLTPaths = ["HLT_HIPhoton20_Core"]
process.hltZMMHI.HLTPaths = ["HLT_HIL2DoubleMu3_Core"]


# Dijet requirement
process.leadingCaloJet = cms.EDFilter( "LargestEtCaloJetSelector",
    src = cms.InputTag( "icPu5CaloJetsL2L3" ),
    filter = cms.bool( False ),
    maxNumber = cms.uint32( 1 )
    )

process.goodLeadingJet = cms.EDFilter("CaloJetSelector",
    src = cms.InputTag("leadingCaloJet"),
    cut = cms.string("et > 130")
    )

process.goodSecondJet = cms.EDFilter("CaloJetSelector",
    src = cms.InputTag("icPu5CaloJetsL2L3"),
    cut = cms.string("et > 50")
    )

process.backToBackDijets = cms.EDProducer("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(False),
    cut = cms.string('abs(deltaPhi(daughter(0).phi,daughter(1).phi)) > 2.5'),
    decay = cms.string("goodLeadingJet goodSecondJet")
    )

process.dijetFilter = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("backToBackDijets"),
    minNumber = cms.uint32(1)
    )

process.backToBackSequence = cms.Sequence(process.leadingCaloJet*process.goodLeadingJet*
                                          process.goodSecondJet*process.backToBackDijets*
                                          process.dijetFilter)

process.particleFlowSequence = cms.Sequence(process.siPixelRecHits * process.siStripMatchedRecHits *
                                            process.heavyIonTracking * process.HiParticleFlowReco)


# Path and EndPath definitions
process.zEESkimPath = cms.Path(process.fullZEESkimSequence)

process.photonSkimPath = cms.Path(process.photonSkimSequence*process.primaryVertexFilter)

process.diJetSkimPath = cms.Path(process.diJetSkimSequence*process.primaryVertexFilter*
                                 process.backToBackSequence*process.particleFlowSequence)

process.zMMSkimPath = cms.Path(process.zMMSkimSequence*process.primaryVertexFilter)

process.SKIMStreamDiJetOutPath = cms.EndPath(process.SKIMStreamDiJet)

process.SKIMStreamPhotonOutPath = cms.EndPath(process.SKIMStreamPhoton)

process.SKIMStreamZEEOutPath = cms.EndPath(process.SKIMStreamZEE)

process.SKIMStreamZMMOutPath = cms.EndPath(process.SKIMStreamZMM)


# Schedule definition
process.schedule = cms.Schedule(process.photonSkimPath,process.zMMSkimPath,process.zEESkimPath,process.diJetSkimPath,process.SKIMStreamDiJetOutPath,process.SKIMStreamPhotonOutPath,process.SKIMStreamZEEOutPath,process.SKIMStreamZMMOutPath)
