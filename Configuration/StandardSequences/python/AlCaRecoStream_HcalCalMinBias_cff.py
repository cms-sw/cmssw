import FWCore.ParameterSet.Config as cms

# last update: $Date$ by $Author$

# HCAL calibration with min.bias
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalMinBias_cff import *

from DQMOffline.Configuration.AlCaRecoDQM_cff import *

pathALCARECOHcalCalMinBias = cms.Path(seqALCARECOHcalCalMinBias)

from Configuration.EventContent.AlCaRecoOutput_cff import *

ALCARECOStreamHcalCalMinBias = cms.FilteredStream(
	responsible = 'Grigory Safronov',
	name = 'ALCARECOHcalCalMinBias',
	paths  = (pathALCARECOHcalCalMinBias),
	content = OutALCARECOHcalCalMinBias.outputCommands,
	selectEvents = OutALCARECOHcalCalMinBias.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)
