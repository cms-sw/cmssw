import FWCore.ParameterSet.Config as cms

# last update: $Date: 2012/03/30 17:12:30 $ by $Author: cerminar $
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



# HCAL calibration with min.bias
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalMinBias_cff import *

from DQMOffline.Configuration.AlCaRecoDQM_cff import *

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


