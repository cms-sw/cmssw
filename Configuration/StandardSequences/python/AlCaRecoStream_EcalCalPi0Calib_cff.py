import FWCore.ParameterSet.Config as cms

# last update: $Date: 2008/08/25 12:50:37 $ by $Author: futyand $

# ECAL calibration with pi0
from Calibration.EcalAlCaRecoProducers.ALCARECOEcalCalPi0Calib_cff import *

from DQMOffline.Configuration.AlCaRecoDQM_cff import *

pathALCARECOEcalCalPi0Calib = cms.Path(seqALCARECOEcalCalPi0Calib*ALCARECOEcalCalPi0CalibDQM)

from Configuration.EventContent.AlCaRecoOutput_cff import *

ALCARECOStreamEcalCalPi0Calib = cms.FilteredStream(
	responsible = 'Vladimir Litvine',
	name = 'ALCARECOEcalCalPi0Calib',
	paths  = (pathALCARECOEcalCalPi0Calib),
	content = OutALCARECOEcalCalPi0Calib.outputCommands,
	selectEvents = OutALCARECOEcalCalPi0Calib.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)
