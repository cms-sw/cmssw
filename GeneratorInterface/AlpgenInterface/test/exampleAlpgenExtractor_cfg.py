###########################################
# Example config file for AlpgenInterface #
###########################################

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

##########################
# Basic process controls #
##########################

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.load("FWCore.MessageService.MessageLogger_cfi")

##########
# Source #
##########

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('file:test.root')
                            )

process.analyzer = cms.EDAnalyzer("AlpgenExtractor",
                                  unwParFile = cms.untracked.string('NEW_unw.par'),
                                  wgtFile = cms.untracked.string('NEW.wgt'),
                                  parFile = cms.untracked.string('NEW.par')
                                  )

process.p = cms.Path(process.analyzer)
