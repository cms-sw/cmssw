import FWCore.ParameterSet.Config as cms

# process
process = cms.Process('L1MuScalesGlobalTagTest')

# number of events and source
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source('EmptySource')

# load and configure modules via Global Tag
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
 
process.GlobalTag.globaltag = 'STARTUP_31X::All'

process.l1muscalestest = cms.EDAnalyzer("L1MuScalesTester")

process.p = cms.Path(process.l1muscalestest)

