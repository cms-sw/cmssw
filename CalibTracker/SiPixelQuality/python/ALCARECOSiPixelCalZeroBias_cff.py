import FWCore.ParameterSet.Config as cms

# Set the HLT paths
import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOSiPixelCalZeroBiasHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    #"regular" Zero Bias
    #HLTPaths = ["HLT_ZeroBias_v*"],
    #eventSetupPathsKey = 'SiPixelCalZeroBias',
    eventSetupPathsKey = 'SiStripCalZeroBias',# use the trigger bit of SiStripCalZeroBias
    throw = False # tolerate triggers stated above, but not available
    )

# Select only events where tracker had HV on (according to DCS bit information)
# AND respective partition is in the run (according to FED information)
import DPGAnalysis.Skims.skim_detstatus_cfi
ALCARECOSiPixelCalZeroBiasDCSFilter = DPGAnalysis.Skims.skim_detstatus_cfi.dcsstatus.clone(
     DetectorType = cms.vstring('BPIX','FPIX'),
     ApplyFilter  = cms.bool(True),
     AndOr        = cms.bool(True),
     DebugOn      = cms.untracked.bool(False)
)

# SiPixelStatus producer
from CalibTracker.SiPixelQuality.SiPixelStatusProducer_cfi import *
# fit as function of lumi sections
siPixelStatusProducer.SiPixelStatusProducerParameters.resetEveryNLumi = 1

# Sequence #
seqALCARECOSiPixelCalZeroBias = cms.Sequence(ALCARECOSiPixelCalZeroBiasHLT*ALCARECOSiPixelCalZeroBiasDCSFilter*siPixelStatusProducer)

## customizations for the pp_on_AA eras
from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
(pp_on_XeXe_2017 | pp_on_AA_2018).toModify(ALCARECOSiPixelCalZeroBiasHLT,
                                           eventSetupPathsKey='SiStripCalZeroBiasHI'
                                           )
