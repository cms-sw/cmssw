import FWCore.ParameterSet.Config as cms

import os 
rootfile_dir = os.environ['CMSSW_BASE'] + '/src/CMGTools/RootTools/data/vertexWeight'
#centraldir = '/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions12/8TeV/PileUp'


#for 52X MC to 2012 ICHEP data set
vertexWeightSummer12MCICHEPData = cms.EDProducer(
    "PileUpWeightProducer",
    verbose = cms.untracked.bool( False ),
    src = cms.InputTag('addPileupInfo'),
    type = cms.int32(2), # 1 = observed , 2= true 
    inputHistMC = cms.string( rootfile_dir + '/Pileup_Summer12MC52X.true.root'),
    inputHistData = cms.string( rootfile_dir + '/Pileup_2012ICHEP_start_196509.true.root' ),
    )

#for 53X MC to 2012 HCP data set

vertexWeightSummer12MC53XICHEPData = cms.EDProducer(
    "PileUpWeightProducer",
    verbose = cms.untracked.bool( False ),
    src = cms.InputTag('addPileupInfo'),
    type = cms.int32(2), # 1 = observed , 2= true 
    inputHistMC = cms.string( rootfile_dir + '/Pileup_Summer12MC53X.true.root'),
    inputHistData = cms.string( rootfile_dir + '/Pileup_2012ICHEP_start_196509.true.root' ),
    )

vertexWeightSummer12MC53XHCPData = cms.EDProducer(
    "PileUpWeightProducer",
    verbose = cms.untracked.bool( False ),
    src = cms.InputTag('addPileupInfo'),
    type = cms.int32(2), # 1 = observed , 2= true 
    inputHistMC = cms.string( rootfile_dir + '/Pileup_Summer12MC53X.true.root'),
    inputHistData = cms.string( rootfile_dir + '/Pileup_2012HCP_190456_203002.true.root' ),
    )

#for 53X MC to first 6/fb of 2012D 
vertexWeightSummer12MC53X2012D6fbData = cms.EDProducer(
    "PileUpWeightProducer",
    verbose = cms.untracked.bool( False ),
    src = cms.InputTag('addPileupInfo'),
    type = cms.int32(2), # 1 = observed , 2= true 
    inputHistMC = cms.string( rootfile_dir + '/Pileup_Summer12MC53X.true.root'),
    inputHistData = cms.string( rootfile_dir + '/Pileup_2012D6fb_203894_207898.true.root' ),
    )


vertexWeightSummer12MC53X2012ABCDData = cms.EDProducer(
    "PileUpWeightProducer",
    verbose = cms.untracked.bool( False ),
    src = cms.InputTag('addPileupInfo'),
    type = cms.int32(2), # 1 = observed , 2= true 
    inputHistMC = cms.string( rootfile_dir + '/Pileup_Summer12MC53X.true.root'),
    inputHistData = cms.string( rootfile_dir + '/Pileup_2012ABCD.true.root' ),
    )

vertexWeightSummer12MC53X2012BCDData = cms.EDProducer(
    "PileUpWeightProducer",
    verbose = cms.untracked.bool( False ),
    src = cms.InputTag('addPileupInfo'),
    type = cms.int32(2), # 1 = observed , 2= true 
    inputHistMC = cms.string( rootfile_dir + '/Pileup_Summer12MC53X.true.root'),
    inputHistData = cms.string( rootfile_dir + '/Pileup_2012BCD.true.root' ),
    )
