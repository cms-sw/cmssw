# Special online streams, Heavy Ions
#
#



import FWCore.ParameterSet.Config as cms

# last update: $Date: 2010/11/18 09:19:15 $ by $Author: argiro $
# $Id: AlCaRecoStream_SpecialsHI_cff.py,v 1.4 2010/11/18 09:19:15 argiro Exp $

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


# last update: $Date: 2010/11/18 09:19:15 $ by $Author: argiro $

# ECAL calibration with pi0
from Calibration.EcalAlCaRecoProducers.ALCARECOEcalCalPi0Calib_cff import *
from DQMOffline.Configuration.AlCaRecoDQMHI_cff import *

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
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalMinBiasHI_cff import *

from DQMOffline.Configuration.AlCaRecoDQMHI_cff import *

pathALCARECOHcalCalMinBias = cms.Path(seqALCARECOHcalCalMinBias*ALCARECOHcalCalPhisymDQM)

from Configuration.EventContent.AlCaRecoOutput_cff import *

ALCARECOStreamHcalCalMinBias = cms.FilteredStream(
        responsible = 'Grigory Safronov',
        name = 'ALCARECOHcalCalMinBias',
        paths  = (pathALCARECOHcalCalMinBias),
        content = OutALCARECOHcalCalMinBiasHI.outputCommands,
        selectEvents = OutALCARECOHcalCalMinBiasHI.SelectEvents,
        dataTier = cms.untracked.string('ALCARECO')
        )
