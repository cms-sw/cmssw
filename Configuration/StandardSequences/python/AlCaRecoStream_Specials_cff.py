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
        name = 'ALCARECOEcalCalEtaCalib',
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
        name = 'ALCARECOEcalCalPi0Calib',
        paths  = (pathALCARECOEcalCalPi0Calib),
        content = OutALCARECOEcalCalPi0Calib.outputCommands,
        selectEvents = OutALCARECOEcalCalPi0Calib.SelectEvents,
        dataTier = cms.untracked.string('ALCARECO')
        )

# HCAL calibration with min.bias
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalMinBias_cff import *

pathALCARECOHcalCalMinBias = cms.Path(seqALCARECOHcalCalMinBias*ALCARECOHcalCalPhisymDQM)

from Configuration.EventContent.AlCaRecoOutput_cff import *

ALCARECOStreamHcalCalMinBias = cms.FilteredStream(
        responsible = 'Grigory Safronov',
        name = 'ALCARECOHcalCalMinBias',
        paths  = (pathALCARECOHcalCalMinBias),
        content = OutALCARECOHcalCalMinBias.outputCommands,
        selectEvents = OutALCARECOHcalCalMinBias.SelectEvents,
        dataTier = cms.untracked.string('ALCARECO')
        )

# HCAL Pedestals
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalPedestal_cff import *

pathALCARECOHcalCalPedestal = cms.Path(seqALCARECOHcalCalPedestal*ALCARECOHcalCalPhisymDQM)

from Configuration.EventContent.AlCaRecoOutput_cff import *

ALCARECOStreamHcalCalPedestal = cms.FilteredStream(
        responsible = 'Olga Kodolova',
        name = 'ALCARECOHcalCalPedestal',
        paths  = (pathALCARECOHcalCalPedestal),
        content = OutALCARECOHcalCalPedestal.outputCommands,
        selectEvents = OutALCARECOHcalCalPedestal.SelectEvents,
        dataTier = cms.untracked.string('ALCARECO')
        )

# AlCaReco for LumiPixel stream
from Calibration.TkAlCaRecoProducers.ALCARECOLumiPixels_cff import *

# FIXME: in case we need to add a DQM step
#from DQMOffline.Configuration.AlCaRecoDQM_cff import *
#pathALCARECOLumiPixels = cms.Path(seqALCARECOLumiPixels*ALCARECOLumiPixelsDQM)

pathALCARECOLumiPixels = cms.Path(seqALCARECOLumiPixels)

from Configuration.EventContent.AlCaRecoOutput_cff import *

ALCARECOStreamLumiPixels = cms.FilteredStream(
        responsible = 'Cerminara Gianluca',
        name = 'ALCARECOLumiPixels',
        paths  = (pathALCARECOLumiPixels),
        content = OutALCARECOLumiPixels.outputCommands,
        selectEvents = OutALCARECOLumiPixels.SelectEvents,
        dataTier = cms.untracked.string('ALCARECO')
        )


