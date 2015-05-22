import FWCore.ParameterSet.Config as cms

import os 
rootfile_dir = os.environ['CMSSW_BASE'] + '/src/CMGTools/RootTools/data/vertexWeight'
centraldir = '/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions11/7TeV/PileUp'


vertexWeightFall11May10ReReco = cms.EDProducer(
    "PileUpWeightProducer",
    verbose = cms.untracked.bool( False ),
    src = cms.InputTag('addPileupInfo'),
    type = cms.int32(1),
    inputHistMC = cms.string( rootfile_dir + '/Pileup_Fall11MC.root'),
    inputHistData = cms.string( centraldir + '/Cert_160404-163869_7TeV_May10ReReco_Collisions11_JSON_v3.pileup_v2.root'),
    )

vertexWeightFall11PromptRecov4 = cms.EDProducer(
    "PileUpWeightProducer",
    verbose = cms.untracked.bool( False ),
    src = cms.InputTag('addPileupInfo'),
    type = cms.int32(1),
    inputHistMC = cms.string( rootfile_dir + '/Pileup_Fall11MC.root'),
    inputHistData = cms.string( centraldir + '/Cert_165088-167913_7TeV_PromptReco_JSON.pileup_v2.root'),
    )

vertexWeightFall1105AugReReco = cms.EDProducer(
    "PileUpWeightProducer",
    verbose = cms.untracked.bool( False ),
    src = cms.InputTag('addPileupInfo'),
    type = cms.int32(1),
    inputHistMC = cms.string( rootfile_dir + '/Pileup_Fall11MC.root'),
    inputHistData = cms.string( centraldir + '/Cert_170249-172619_7TeV_ReReco5Aug_Collisions11_JSON_v2.pileup_v2.root'),
    )

###PromptRecov6 same as Oct3ReReco
vertexWeightFall11PromptRecov6 = cms.EDProducer(
    "PileUpWeightProducer",
    verbose = cms.untracked.bool( False ),
    src = cms.InputTag('addPileupInfo'),
    type = cms.int32(1),
    inputHistMC = cms.string( rootfile_dir + '/Pileup_Fall11MC.root'),
    inputHistData = cms.string( centraldir + '/Cert_172620-173692_PromptReco_JSON.pileup_v2.root'),
    )


vertexWeightFall112011B = cms.EDProducer(
    "PileUpWeightProducer",
    verbose = cms.untracked.bool( False ),
    src = cms.InputTag('addPileupInfo'),
    type = cms.int32(1),
    inputHistMC = cms.string( rootfile_dir + '/Pileup_Fall11MC.root'),
    inputHistData = cms.string( rootfile_dir + '/Pileup_2011B.pileup.root'),
    )


vertexWeightFall11EPSJul8 = cms.EDProducer(
    "PileUpWeightProducer",
    verbose = cms.untracked.bool( False ),
    src = cms.InputTag('addPileupInfo'),
    type = cms.int32(1),
    inputHistMC = cms.string( rootfile_dir + '/Pileup_Fall11MC.root'),
    inputHistData = cms.string( centraldir + '/Pileup_2011_EPS_8_jul.root'),
    )

vertexWeightFall11LeptonPhoton = cms.EDProducer(
    "PileUpWeightProducer",
    verbose = cms.untracked.bool( False ),
    src = cms.InputTag('addPileupInfo'),
    type = cms.int32(1),
    inputHistMC = cms.string( rootfile_dir + '/Pileup_Fall11MC.root'),
    inputHistData = cms.string( centraldir + '/Pileup_2011_to_172802_LP_LumiScale.root'),
    )


#weight covering May10ReReco + PromptReco-v4 + 05AugReReco + Prompt-v6  = 2011A
vertexWeightFall112invfb = cms.EDProducer(
    "PileUpWeightProducer",
    verbose = cms.untracked.bool( False ),
    src = cms.InputTag('addPileupInfo'),
    type = cms.int32(1),
    inputHistMC = cms.string( rootfile_dir + '/Pileup_Fall11MC.root'),
    inputHistData = cms.string( rootfile_dir + '/Pileup_160404-173692_2.1invfb.pileup.root' ),
    )

#full 2011 data 2011A + 2011B
vertexWeightFall112011AB = cms.EDProducer(
    "PileUpWeightProducer",
    verbose = cms.untracked.bool( False ),
    src = cms.InputTag('addPileupInfo'),
    type = cms.int32(1),
    inputHistMC = cms.string( rootfile_dir + '/Pileup_Fall11MC.root'),
    inputHistData = cms.string( rootfile_dir + '/Pileup_160404-180252_4.6invfb.pileup.root' ),
    )
