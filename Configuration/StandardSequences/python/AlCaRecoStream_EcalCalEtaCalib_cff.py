import FWCore.ParameterSet.Config as cms

# last update: $Date: 2009/02/28 15:56:07 $ by $Author: dlange $

# ECAL calibration with eta
from Calibration.EcalAlCaRecoProducers.ALCARECOEcalCalEtaCalib_cff import *
from DQMOffline.Configuration.AlCaRecoDQM_cff import *

pathALCARECOEcalCalEtaCalib = cms.Path(seqALCARECOEcalCalEtaCalib*ALCARECOEcalCalEtaCalibDQM)

from Configuration.EventContent.AlCaRecoOutput_cff import *

ALCARECOStreamEcalCalEtaCalib = cms.FilteredStream(
	responsible = 'Vladimir Litvine',
	name = 'ALCARECOEcalCalEtaCalib',
	paths  = (pathALCARECOEcalCalEtaCalib),
	content = OutALCARECOEcalCalEtaCalib.outputCommands,
	selectEvents = OutALCARECOEcalCalEtaCalib.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)
