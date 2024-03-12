import FWCore.ParameterSet.Config as cms

HcalGeometryFromDBEP = cms.ESProducer("HcalGeometryFromDBEP",
                                      applyAlignment = cms.bool(False)
                                      )

HcalAlignmentEP = cms.ESProducer("HcalAlignmentEP")

# foo bar baz
# Yzl1TQ5hKXQ4M
# xK940tOA5O56s
