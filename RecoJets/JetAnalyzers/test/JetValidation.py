# PYTHON configuration file for class: JetValidation
# Description:  Some Basic validation plots for jets.
# Author: K. Kousouris
# Date:  27 - August - 2008
import FWCore.ParameterSet.Config as cms

process = cms.Process("Ana")
process.load("FWCore.MessageService.MessageLogger_cfi")
#############   Set the number of events #############
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)
#############   Define the source file ###############
#process.load("RecoJets.JetAnalyzers.RelValQCD_Pt_80_120_cfi")
process.source = cms.Source("PoolSource",fileNames = cms.untracked.vstring('/store/relval/CMSSW_2_1_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V6_10TeV_v7/0000/044A7A03-BE6E-DD11-B3C0-000423D98B08.root'))
#############   jet tracks association ###############
from RecoJets.JetAssociationProducers.j2tParametersVX_cfi import *     
process.JetTracksAssociatorAtVertex = cms.EDProducer("JetTracksAssociatorAtVertex",
     j2tParametersVX,
     jets         = cms.InputTag("sisCone7CaloJets"),
     coneSize     = cms.double(0.7)
)
#############   Define the process ###################
process.validation = cms.EDAnalyzer("JetValidation",
    PtMin               = cms.double(5.),
    dRmatch             = cms.double(0.25),
    MCarlo              = cms.bool(True),
    calAlgo             = cms.string('sisCone7CaloJets'),
    genAlgo             = cms.string('sisCone7GenJets'),
    jetTracksAssociator = cms.string('JetTracksAssociatorAtVertex'), 
    histoFileName       = cms.string('JetValidation.root'),
    Njets               = cms.int32(100)
)
#############   Path       ###########################
process.p = cms.Path(process.JetTracksAssociatorAtVertex * process.validation)
#############   Format MessageLogger #################
process.MessageLogger.cerr.FwkReport.reportEvery = 10

