import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

dataIntegrityTest = DQMEDHarvester("DTDataIntegrityTest",
                                   diagnosticPrescale = cms.untracked.int32(1),
				   checkUros = cms.untracked.bool(False) 
)

from Configuration.Eras.Modifier_run2_DT_2018_cff import run2_DT_2018
run2_DT_2018.toModify(dataIntegrityTest,checkUros=cms.untracked.bool(True))

