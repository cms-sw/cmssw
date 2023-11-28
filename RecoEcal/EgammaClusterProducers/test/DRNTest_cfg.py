import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

from Configuration.ProcessModifiers.enableSonicTriton_cff import enableSonicTriton
process = cms.Process("Demo",enableSonicTriton)

process.load("HeterogeneousCore.SonicTriton.TritonService_cff")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )
process.options.numberOfThreads = cms.untracked.uint32(4)
process.options.numberOfStreams = cms.untracked.uint32(4)
process.options.TryToContinue = cms.untracked.vstring('ProductNotFound')

process.TritonService.verbose = False
process.TritonService.fallback.verbose = False
process.TritonService.fallback.useDocker = False
process.TritonService.fallback.useGPU = False

process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(100)
#process.MessageLogger.suppressWarning = cms.untracked.vstring('DRNProducerEB', 'DRNProducerEE')

process.source = cms.Source("PoolSource",
                                # replace 'myfile.root' with the source file you want to use
                                fileNames = cms.untracked.vstring(
'/store/mc/RunIISummer20UL18RECO/DoubleElectron_Pt-1To300-gun/AODSIM/FlatPU0to70EdalIdealGT_EdalIdealGT_106X_upgrade2018_realistic_v11_L1v1_EcalIdealIC-v2/270000/4CDD9457-E14C-D84A-9BD4-3140CB6AEEB6.root',
'/store/mc/RunIISummer20UL18RECO/DoubleElectron_Pt-1To300-gun/AODSIM/FlatPU0to70EdalIdealGT_EdalIdealGT_106X_upgrade2018_realistic_v11_L1v1_EcalIdealIC-v2/270000/D3A06456-7F7D-C940-9E56-0DCA06B3ECC9.root',
'/store/mc/RunIISummer20UL18RECO/DoubleElectron_Pt-1To300-gun/AODSIM/FlatPU0to70EdalIdealGT_EdalIdealGT_106X_upgrade2018_realistic_v11_L1v1_EcalIdealIC-v2/270000/477810A1-AE5C-6A49-8067-35776F0C78B6.root',
'/store/mc/RunIISummer20UL18RECO/DoubleElectron_Pt-1To300-gun/AODSIM/FlatPU0to70EdalIdealGT_EdalIdealGT_106X_upgrade2018_realistic_v11_L1v1_EcalIdealIC-v2/270000/99F8FA99-B120-BF40-BD67-7825696B9E78.root'

                )
                            )

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '106X_upgrade2018_realistic_v11_Ecal5', '')
#from RecoEcal.EgammaClusterProducers.SCEnergyCorrectorDRNProducer_cfi import *
from RecoEcal.EgammaClusterProducers.SCEnergyCorrectorDRNProducer_cfi import *

process.DRNProducerEB = DRNProducerEB
process.DRNProducerEE = DRNProducerEE

from PhysicsTools.SelectorUtils.tools.vid_id_tools import *
dataFormat = DataFormat.AOD
switchOnVIDElectronIdProducer(process, dataFormat)

# define which IDs we want to produce
my_id_modules = ['RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Fall17_94X_V2_cff']

for idmod in my_id_modules:
        setupAllVIDIdsInModule(process, idmod, setupVIDElectronSelection)

process.nTuplelize = cms.EDAnalyzer('DRNTestNTuplizer',
        vertexCollection = cms.InputTag('offlinePrimaryVertices'),
        rhoFastJet = cms.InputTag("fixedGridRhoFastjetAll"),
        pileupInfo = cms.InputTag("addPileupInfo"),
        electrons = cms.InputTag("gedGsfElectrons"),
        genParticles = cms.InputTag("genParticles"),
        #Cut Based Id
        eleLooseIdMap = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V2-loose"),
        eleMediumIdMap = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V2-medium"),
        eleTightIdMap = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V2-tight"),

        tracks = cms.InputTag("globalTracks"),
        SkipEvent = cms.untracked.vstring('ProductNotFound')
        )

process.TFileService = cms.Service("TFileService",
     fileName = cms.string("nTupleMC.root"),
      closeFileFast = cms.untracked.bool(True)
  )

process.load( "HLTrigger.Timer.FastTimerService_cfi" )

#process.p = cms.Path(process.DRNProducerEB*process.egmGsfElectronIDSequence*process.nTuplelize)

#process.p = cms.Path(process.DRNProducerEB*process.DRNProducerEE*process.egmGsfElectronIDSequence*process.nTuplelize)
process.p = cms.Path(process.DRNProducerEB*process.DRNProducerEE)
