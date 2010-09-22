import FWCore.ParameterSet.Config as cms

# last update: $Date: 2008/08/25 11:36:00 $ by $Author: futyand $

# ECAL calibration with phi symmetry 
from Calibration.EcalAlCaRecoProducers.ALCARECOEcalCalPhiSym_cff import *

from DQMOffline.Configuration.AlCaRecoDQMHI_cff import *

pathALCARECOEcalCalPhiSym = cms.Path(seqALCARECOEcalCalPhiSym*ALCARECOEcalCalPhisymDQM)

from Configuration.EventContent.AlCaRecoOutput_cff import *

ALCARECOStreamEcalCalPhiSym = cms.FilteredStream(
	responsible = 'Stefano Argiro',
	name = 'ALCARECOEcalCalPhiSym',
	paths  = (pathALCARECOEcalCalPhiSym),
	content = OutALCARECOEcalCalPhiSym.outputCommands,
	selectEvents = OutALCARECOEcalCalPhiSym.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)
