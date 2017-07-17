import FWCore.ParameterSet.Config as cms

# last update: $Date: 2010/09/27 11:38:30 $ by $Author: argiro $
#
# Special online streams
#
#


# ECAL calibration with eta
from Calibration.EcalAlCaRecoProducers.ALCARECOEcalCalEtaCalib_cff import *
from DQMOffline.Configuration.AlCaRecoDQM_cff import *

pathALCARECOEcalCalEtaCalib = cms.Path(seqALCARECOEcalCalEtaCalib*ALCARECOEcalCalEtaCalibDQM)

from Configuration.EventContent.AlCaRecoOutput_cff import *

ALCARECOStreamEcalCalEtaCalib = cms.FilteredStream(
        responsible = 'Vladimir Litvine',
        name = 'EcalCalEtaCalib',
        paths  = (pathALCARECOEcalCalEtaCalib),
        content = OutALCARECOEcalCalEtaCalib.outputCommands,
        selectEvents = OutALCARECOEcalCalEtaCalib.SelectEvents,
        dataTier = cms.untracked.string('ALCARECO')
        )


# ECAL calibration with pi0
from Calibration.EcalAlCaRecoProducers.ALCARECOEcalCalPi0Calib_cff import *

pathALCARECOEcalCalPi0Calib = cms.Path(seqALCARECOEcalCalPi0Calib*ALCARECOEcalCalPi0CalibDQM)

from Configuration.EventContent.AlCaRecoOutput_cff import *

ALCARECOStreamEcalCalPi0Calib = cms.FilteredStream(
        responsible = 'Vladimir Litvine',
        name = 'EcalCalPi0Calib',
        paths  = (pathALCARECOEcalCalPi0Calib),
        content = OutALCARECOEcalCalPi0Calib.outputCommands,
        selectEvents = OutALCARECOEcalCalPi0Calib.SelectEvents,
        dataTier = cms.untracked.string('ALCARECO')
        )

# HCAL calibration with min.bias
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalMinBias_cff import *

pathALCARECOHcalCalMinBias = cms.Path(seqALCARECOHcalCalMinBiasDigi*seqALCARECOHcalCalMinBias*ALCARECOHcalCalPhisymDQM)

from Configuration.EventContent.AlCaRecoOutput_cff import *

ALCARECOStreamHcalCalMinBias = cms.FilteredStream(
        responsible = 'Grigory Safronov',
        name = 'HcalCalMinBias',
        paths  = (pathALCARECOHcalCalMinBias),
        content = OutALCARECOHcalCalMinBias.outputCommands,
        selectEvents = OutALCARECOHcalCalMinBias.SelectEvents,
        dataTier = cms.untracked.string('ALCARECO')
        )

# HCAL Pedestals
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalPedestal_cff import *

pathALCARECOHcalCalPedestal = cms.Path(seqALCARECOHcalCalPedestalDigi*seqALCARECOHcalCalPedestal*ALCARECOHcalCalPhisymDQM)

from Configuration.EventContent.AlCaRecoOutput_cff import *

ALCARECOStreamHcalCalPedestal = cms.FilteredStream(
        responsible = 'Olga Kodolova',
        name = 'HcalCalPedestal',
        paths  = (pathALCARECOHcalCalPedestal),
        content = OutALCARECOHcalCalPedestal.outputCommands,
        selectEvents = OutALCARECOHcalCalPedestal.SelectEvents,
        dataTier = cms.untracked.string('ALCARECO')
        )

# AlCaReco for LumiPixel stream
from Calibration.LumiAlCaRecoProducers.ALCARECOLumiPixels_cff import *
from Calibration.LumiAlCaRecoProducers.ALCARECOAlCaPCCZeroBias_cff import *
from Calibration.LumiAlCaRecoProducers.ALCARECOAlCaPCCRandom_cff import *

# FIXME: in case we need to add a DQM step
#from DQMOffline.Configuration.AlCaRecoDQM_cff import *
#pathALCARECOLumiPixels = cms.Path(seqALCARECOLumiPixels*ALCARECOLumiPixelsDQM)

pathALCARECOLumiPixels      = cms.Path(seqALCARECOLumiPixels)
pathALCARECOAlCaPCCZeroBias = cms.Path(seqALCARECOAlCaPCCZeroBias)
pathALCARECOAlCaPCCRandom   = cms.Path(seqALCARECOAlCaPCCRandom)

from Configuration.EventContent.AlCaRecoOutput_cff import *

ALCARECOStreamLumiPixels = cms.FilteredStream(
        responsible = 'Cerminara Gianluca',
        name = 'LumiPixels',
        paths  = (pathALCARECOLumiPixels),
        content = OutALCARECOLumiPixels.outputCommands,
        selectEvents = OutALCARECOLumiPixels.SelectEvents,
        dataTier = cms.untracked.string('ALCARECO')
        )

ALCARECOStreamAlCaPCCZeroBias = cms.FilteredStream(
	      responsible = 'Chris Palmer',
	      name = 'AlCaPCCZeroBias',
	      paths  = (pathALCARECOAlCaPCCZeroBias),
	      content = OutALCARECOAlCaPCCZeroBias.outputCommands,
	      selectEvents = OutALCARECOAlCaPCCZeroBias.SelectEvents,
	      dataTier = cms.untracked.string('ALCARECO')
	      )

ALCARECOStreamAlCaPCCRandom = cms.FilteredStream(
	      responsible = 'Chris Palmer',
	      name = 'AlCaPCCRandom',
	      paths  = (pathALCARECOAlCaPCCRandom),
	      content = OutALCARECOAlCaPCCRandom.outputCommands,
	      selectEvents = OutALCARECOAlCaPCCRandom.SelectEvents,
	      dataTier = cms.untracked.string('ALCARECO')
	      )


