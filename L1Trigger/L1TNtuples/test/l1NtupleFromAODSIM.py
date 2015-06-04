import FWCore.ParameterSet.Config as cms

# make ntuples from RECO (ie. remove RAW)

from L1TriggerDPG.L1Ntuples.l1Ntuple_cfg import *

# remove stuff that depends on RAW data
process.p.remove(process.gtDigis)
process.p.remove(process.gtEvmDigis)
process.p.remove(process.gctDigis)
process.p.remove(process.dttfDigis)
process.p.remove(process.csctfDigis)
process.p.remove(process.l1extraParticles)

process.l1NtupleProducer.gmtSource            = cms.InputTag("none")
process.l1NtupleProducer.gtEvmSource          = cms.InputTag("none")
process.l1NtupleProducer.gtSource             = cms.InputTag("none")
process.l1NtupleProducer.gctCentralJetsSource = cms.InputTag("none")
process.l1NtupleProducer.gctNonIsoEmSource    = cms.InputTag("none")
process.l1NtupleProducer.gctForwardJetsSource = cms.InputTag("none")
process.l1NtupleProducer.gctIsoEmSource       = cms.InputTag("none")
process.l1NtupleProducer.gctEnergySumsSource  = cms.InputTag("none")
process.l1NtupleProducer.gctTauJetsSource     = cms.InputTag("none")
process.l1NtupleProducer.rctSource            = cms.InputTag("none")
process.l1NtupleProducer.dttfSource           = cms.InputTag("none")
process.l1NtupleProducer.csctfTrkSource       = cms.InputTag("none")
process.l1NtupleProducer.csctfLCTSource       = cms.InputTag("none")
process.l1NtupleProducer.csctfStatusSource    = cms.InputTag("none")
process.l1NtupleProducer.csctfDTStubsSource   = cms.InputTag("none")

# remove RECO temporarily
process.p.remove(process.l1RecoTreeProducer)
process.p.remove(process.l1MuonRecoTreeProducer)

# PU re-weighting
process.l1NtupleProducer.puMCFile = cms.untracked.string("PUHistS10.root")
process.l1NtupleProducer.puDataFile = cms.untracked.string("")   # won't work without a filename here!
process.l1NtupleProducer.puMCHist = cms.untracked.string("pileup")
process.l1NtupleProducer.puDataHist = cms.untracked.string("pileup") 



# job options

process.GlobalTag.globaltag = 'GR_R_52_V7::All'

SkipEvent = cms.untracked.vstring('ProductNotFound')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

readFiles = cms.untracked.vstring(
    'file:/gpfs_phys/storm/cms/mc/Summer12/ZJetsToNuNu_100_HT_200_TuneZ2Star_8TeV_madgraph/AODSIM/PU_S7_START52_V9-v1/0000/00367B90-55AD-E111-B03E-E0CB4E1A11A2.root'
    )
secFiles = cms.untracked.vstring()
process.source = cms.Source ("PoolSource",
                             fileNames = readFiles,
                             secondaryFileNames = secFiles
                             )


