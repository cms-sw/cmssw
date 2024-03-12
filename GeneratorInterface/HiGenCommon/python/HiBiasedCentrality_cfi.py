import FWCore.ParameterSet.Config as cms

hiBiasedCentrality = cms.EDFilter('HiCentralityBiasFilter',
                                  function = cms.string("gaus"),
                                  parameters = cms.vdouble(1,2.42651,3.79481)
                                  )


# foo bar baz
# JRg0N68GVmKFE
# ioWBQwnYOgGKn
