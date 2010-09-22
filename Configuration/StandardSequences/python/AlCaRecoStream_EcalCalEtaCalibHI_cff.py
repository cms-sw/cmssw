import FWCore.ParameterSet.Config as cms

# last update: $Date: 2009/03/26 12:03:01 $ by $Author: argiro $

# ECAL calibration with eta
from Calibration.EcalAlCaRecoProducers.ALCARECOEcalCalEtaCalib_cff import *
from DQMOffline.Configuration.AlCaRecoDQMHI_cff import *

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
