import FWCore.ParameterSet.Config as cms

stat = cms.EDAnalyzer("SiStripQualityStatistics",
                      dataLabel = cms.untracked.string("unbiased"),
                      SaveTkHistoMap = cms.untracked.bool(False),
                      TkMapFileName = cms.untracked.string("TkMapBadComponents.png")  #available filetypes: .pdf .png .jpg .svg
                      )

