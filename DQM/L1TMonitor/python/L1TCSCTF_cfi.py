import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tCsctf = DQMEDAnalyzer("L1TCSCTF",
    gmtProducer = cms.InputTag("l1GtUnpack"),

    statusProducer = cms.InputTag("csctfDigis"),
    outputFile = cms.untracked.string(''),
    lctProducer = cms.InputTag("csctfDigis"),
    verbose = cms.untracked.bool(False),
    gangedME11a = cms.untracked.bool(True), ## Run2: False; Run1: True
    trackProducer = cms.InputTag("csctfDigis"),
    mbProducer = cms.InputTag("csctfDigis:DT"),
    DQMStore = cms.untracked.bool(True),
    disableROOToutput = cms.untracked.bool(True)
)

#
# Make changes for running in Run 2
#
from Configuration.Eras.Modifier_run2_common_cff import run2_common
run2_common.toModify( l1tCsctf, gangedME11a = False )
