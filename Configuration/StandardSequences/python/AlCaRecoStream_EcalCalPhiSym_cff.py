import FWCore.ParameterSet.Config as cms

# last update: $Date$ by $Author$

# ECAL calibration with phi symmetry 
from Calibration.EcalAlCaRecoProducers.ALCARECOEcalCalPhiSym_cff import *

from DQMOffline.Configuration.AlCaRecoDQM_cff import *

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
