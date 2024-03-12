
import FWCore.ParameterSet.Config as cms


from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
DQMMessageLogger = DQMEDAnalyzer('DQMMessageLogger',
                             Categories = cms.vstring(),
                             Directory = cms.string("MessageLogger")
                             )


# foo bar baz
# 8pLP0pBs4GewX
# GtB3xQUGayhr8
