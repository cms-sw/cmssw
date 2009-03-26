import FWCore.ParameterSet.Config as cms

# last update: $Date: 2008/08/25 11:36:00 $ by $Author: futyand $

# HCAL calibration with isolated tracks
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalIsoTrk_cff import *

from DQMOffline.Configuration.AlCaRecoDQM_cff import *

pathALCARECOHcalCalIsoTrk = cms.Path(seqALCARECOHcalCalIsoTrk*ALCARECOHcalCalIsoTrackDQM)

from Configuration.EventContent.AlCaRecoOutput_cff import *

ALCARECOStreamHcalCalIsoTrk = cms.FilteredStream(
	responsible = 'Grigory Safronov',
	name = 'ALCARECOHcalCalIsoTrk',
	paths  = (pathALCARECOHcalCalIsoTrk),
	content = OutALCARECOHcalCalIsoTrk.outputCommands,
	selectEvents = OutALCARECOHcalCalIsoTrk.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)
