import FWCore.ParameterSet.Config as cms
from  PhysicsTools.NanoAOD.common_cff import *

hcalDetIdTable = cms.EDProducer("HcalDetIdTableProducer")
hcalDetIdTableTask = cms.Task(hcalDetIdTable)
hcalDetIdTableSeq = cms.Sequence(hcalDetIdTable)
# foo bar baz
# 6lERSJs7wFCGr
