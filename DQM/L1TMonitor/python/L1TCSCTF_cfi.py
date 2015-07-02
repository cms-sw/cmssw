import FWCore.ParameterSet.Config as cms

l1tCsctf = cms.EDAnalyzer("L1TCSCTF",
    gmtProducer = cms.InputTag("gtDigis"),

    statusProducer = cms.InputTag("csctfDigis"),
    outputFile = cms.untracked.string(''),
    lctProducer = cms.InputTag("csctfDigis"),
    verbose = cms.untracked.bool(False),
    gangedME11a = cms.untracked.bool(False), ## Run2: False; Run1: True
    trackProducer = cms.InputTag("csctfDigis"),
    mbProducer = cms.InputTag("csctfDigis:DT"),
    DQMStore = cms.untracked.bool(True),
    disableROOToutput = cms.untracked.bool(True)
)

#
# Make changes for running in Run 2
#
# This is not working.
#from Configuration.StandardSequences.Eras import eras
#eras.run2_common.toModify( l1tCsctf, gangedME11a = False )
