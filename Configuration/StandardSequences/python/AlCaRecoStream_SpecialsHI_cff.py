# Special online streams, Heavy Ions
#
#



import FWCore.ParameterSet.Config as cms

# last update: $Date: 2010/11/18 09:19:15 $ by $Author: argiro $
# $Id: AlCaRecoStream_SpecialsHI_cff.py,v 1.4 2010/11/18 09:19:15 argiro Exp $

# ECAL calibration with eta
from Calibration.EcalAlCaRecoProducers.ALCARECOEcalCalEtaCalib_cff import *
from DQMOffline.Configuration.AlCaRecoDQMHI_cff import *

# Short-term workaround to preserve the "run for every event" while removing the use of convertToUnscheduled()
# To be reverted in a subsequent PR
pathALCARECOEcalCalEtaCalibTask = cms.Task(ALCARECOEcalCalEtaCalibDQM)
pathALCARECOEcalCalEtaCalib = cms.Path(seqALCARECOEcalCalEtaCalib, pathALCARECOEcalCalEtaCalibTask)

from Configuration.EventContent.AlCaRecoOutput_cff import *

ALCARECOStreamEcalCalEtaCalib = cms.FilteredStream(
        responsible = 'Vladimir Litvine',
        name = 'EcalCalEtaCalib',
        paths  = (pathALCARECOEcalCalEtaCalib),
        content = OutALCARECOEcalCalEtaCalib.outputCommands,
        selectEvents = OutALCARECOEcalCalEtaCalib.SelectEvents,
        dataTier = cms.untracked.string('ALCARECO')
        )


# last update: $Date: 2010/11/18 09:19:15 $ by $Author: argiro $

# ECAL calibration with pi0
from Calibration.EcalAlCaRecoProducers.ALCARECOEcalCalPi0Calib_cff import *
from DQMOffline.Configuration.AlCaRecoDQMHI_cff import *

# Short-term workaround to preserve the "run for every event" while removing the use of convertToUnscheduled()
# To be reverted in a subsequent PR
pathALCARECOEcalCalPi0CalibDQM = cms.Task(ALCARECOEcalCalPi0CalibDQM)
pathALCARECOEcalCalPi0Calib = cms.Path(seqALCARECOEcalCalPi0Calib, pathALCARECOEcalCalPi0CalibDQM)

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
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalMinBiasHI_cff import *

from DQMOffline.Configuration.AlCaRecoDQMHI_cff import *

# Short-term workaround to preserve the "run for every event" while removing the use of convertToUnscheduled()
# To be reverted in a subsequent PR
pathALCARECOHcalCalMinBiasTask = cms.Task(ALCARECOHcalCalPhisymDQM)
pathALCARECOHcalCalMinBias = cms.Path(seqALCARECOHcalCalMinBiasDigi*seqALCARECOHcalCalMinBias, pathALCARECOHcalCalMinBiasTask)

from Configuration.EventContent.AlCaRecoOutput_cff import *

ALCARECOStreamHcalCalMinBias = cms.FilteredStream(
        responsible = 'Grigory Safronov',
        name = 'HcalCalMinBias',
        paths  = (pathALCARECOHcalCalMinBias),
        content = OutALCARECOHcalCalMinBiasHI.outputCommands,
        selectEvents = OutALCARECOHcalCalMinBiasHI.SelectEvents,
        dataTier = cms.untracked.string('ALCARECO')
        )
