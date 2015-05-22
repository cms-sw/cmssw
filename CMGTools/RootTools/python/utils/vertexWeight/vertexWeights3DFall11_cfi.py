import FWCore.ParameterSet.Config as cms

import os 
rootfile_dir = os.environ['CMSSW_BASE'] + '/src/CMGTools/RootTools/data/vertexWeight'
centraldir = '/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions11/7TeV/PileUp'

vertexWeight3DFall11May10ReReco = cms.EDProducer(
    "PileUpWeight3DProducer",
    verbose = cms.untracked.bool( False ),
    inputHistMC = cms.string( rootfile_dir + '/Pileup3D_Fall11MC.root'),
    inputHistData = cms.string( centraldir + '/Cert_160404-163869_7TeV_May10ReReco_Collisions11_JSON_v3.pileupTruth_v2_finebin.root'),
    )

vertexWeight3DFall11PromptRecov4 = cms.EDProducer(
    "PileUpWeight3DProducer",
    verbose = cms.untracked.bool( False ),
    inputHistMC = cms.string( rootfile_dir + '/Pileup3D_Fall11MC.root'),
    inputHistData = cms.string( centraldir + '/Cert_165088-167913_7TeV_PromptReco_JSON.pileupTruth_v2_finebin.root'),
    )

vertexWeight3DFall1105AugReReco = cms.EDProducer(
    "PileUpWeight3DProducer",
    verbose = cms.untracked.bool( False ),
    inputHistMC = cms.string( rootfile_dir + '/Pileup3D_Fall11MC.root'),
    inputHistData = cms.string( centraldir + '/Cert_170249-172619_7TeV_ReReco5Aug_Collisions11_JSON_v2.pileupTruth_v2_finebin.root'),
    )

#same as 0ct3ReReco
vertexWeight3DFall11PromptRecov6 = cms.EDProducer(
    "PileUpWeight3DProducer",
    verbose = cms.untracked.bool( False ),
    inputHistMC = cms.string( rootfile_dir + '/Pileup3D_Fall11MC.root'),
    inputHistData = cms.string( centraldir + '/Cert_172620-173692_PromptReco_JSON.pileupTruth_v2_finebin.root'),
    )

#weight covers May10ReReco + PromptReco-v4 + 05AugReReco + Prompt-v6 = 2011A
vertexWeight3DFall112invfb = cms.EDProducer(
    "PileUpWeight3DProducer",
    verbose = cms.untracked.bool( False ),
    inputHistMC = cms.string( rootfile_dir + '/Pileup3D_Fall11MC.root'),
    inputHistData = cms.string( rootfile_dir + '/Pileup3D_160404-173692_2.1invfb.pileup.root' ),
    )

vertexWeight3DFall112011B = cms.EDProducer(
    "PileUpWeight3DProducer",
    verbose = cms.untracked.bool( False ),
    inputHistMC = cms.string( rootfile_dir + '/Pileup3D_Fall11MC.root'),
    inputHistData = cms.string( rootfile_dir + '/Pileup3D_2011B.pileup.root'),
    )

vertexWeight3DFall112011AB = cms.EDProducer(
    "PileUpWeight3DProducer",
    verbose = cms.untracked.bool( False ),
    inputHistMC = cms.string( rootfile_dir + '/Pileup3D_Fall11MC.root'),
    inputHistData = cms.string( rootfile_dir + '/Pileup3D_160404-180252_4.6invfb.pileup.root' ),
    )

