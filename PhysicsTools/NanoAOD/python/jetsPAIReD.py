import FWCore.ParameterSet.Config as cms
from  PhysicsTools.NanoAOD.common_cff import *

pairedJetTable = cms.EDProducer("PAIReDONNXJetTagsProducer")

pairedJetTableTask = cms.Task(pairedJetTable)

pairedJetTableMC = cms.EDProducer("PAIReDONNXJetTagsProducer")

pairedJetTableMCTask = cms.Task(pairedJetTableMC)
