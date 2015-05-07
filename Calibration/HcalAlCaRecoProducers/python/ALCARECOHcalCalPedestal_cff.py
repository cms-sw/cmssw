import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL minbias:
#------------------------------------------------

from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalMinBias_cff import *
hcalDigiAlCaMB.InputLabel = 'hltHcalCalibrationRaw'
#from EventFilter.HcalRawToDigi.HcalRawToDigi_cfi import *
#hcalDigis.InputLabel = 'hltHcalCalibrationRaw'
from RecoLocalCalo.Configuration.hcalLocalRecoNZS_cff import *

hbherecoMB.firstSample = 0
hbherecoMB.samplesToAdd = 4
hbherecoMB.digiLabel = 'hcalDigiAlCaMB'

hfrecoMB.firstSample = 0
hfrecoMB.samplesToAdd = 2
hfrecoMB.digiLabel = 'hcalDigiAlCaMB'

horecoMB.firstSample = 0
horecoMB.samplesToAdd = 4
horecoMB.digiLabel = 'hcalDigiAlCaMB'


import EventFilter.HcalRawToDigi.HcalCalibTypeFilter_cfi
hcalCalibPedestal = EventFilter.HcalRawToDigi.HcalCalibTypeFilter_cfi.hcalCalibTypeFilter.clone(
#  InputLabel = cms.string('rawDataCollector'), 
  InputLabel = cms.string('hltHcalCalibrationRaw::HLT'),
#  InputLabel = cms.InputTag("hltEcalCalibrationRaw","","HLT"),
  CalibTypes    = cms.vint32( 1 ),
  FilterSummary = cms.untracked.bool( False )
)

seqALCARECOHcalCalPedestal = cms.Sequence(hcalCalibPedestal*hcalDigiAlCaMB*gtDigisAlCaMB*hcalLocalRecoSequenceNZS*hbherecoNoise*hfrecoNoise*horecoNoise)
