import FWCore.ParameterSet.Config as cms

# last update: $Date$ by $Author$

# HCAL calibration with isolated tracks
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalIsoTrk_cff import *

from DQMOffline.Configuration.AlCaRecoDQM_cff import *

pathALCARECOHcalCalIsoTrk = cms.Path(seqALCARECOHcalCalIsoTrk)

from Configuration.EventContent.AlCaRecoOutput_cff import *

ALCARECOStreamHcalCalIsoTrk = cms.FilteredStream(
	responsible = 'Grigory Safronov',
	name = 'ALCARECOHcalCalIsoTrk',
	paths  = (pathALCARECOHcalCalIsoTrk),
	content = OutALCARECOHcalCalIsoTrk.outputCommands,
	selectEvents = OutALCARECOHcalCalIsoTrk.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)
