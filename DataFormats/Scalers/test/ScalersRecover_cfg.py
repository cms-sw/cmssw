#
#  Example python configuration file for running ScalersRecover
#  This file examines run 136035            (w.badgett)
#

import FWCore.ParameterSet.Config as cms

process = cms.Process("ScalersRecover")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

#
# Query the DBS to find out the file names, given a run number and
#   the MinBias RECO stream.   Make sure that the file containing
#   lumi sections 1 and 2 comes first in the list!   Note that the
#   trigger scalers record is generally one lumi section out of phase
#   with the event data.
#
process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
    '/store/data/Run2010A/MinimumBias/RECO/v1/000/136/035/FA28577B-C365-DF11-952F-001617E30CC8.root',
    '/store/data/Run2010A/MinimumBias/RECO/v1/000/136/035/DCA60B3F-A565-DF11-A9B6-0030487CAF5E.root',
    '/store/data/Run2010A/MinimumBias/RECO/v1/000/136/035/06F7F82C-AA65-DF11-B3AC-0030487A3DE0.root'
  )
)

process.ScalersRecover = cms.EDAnalyzer('ScalersRecover')


process.p = cms.Path(process.ScalersRecover)
