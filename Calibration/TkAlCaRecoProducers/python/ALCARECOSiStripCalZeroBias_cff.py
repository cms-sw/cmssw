import FWCore.ParameterSet.Config as cms

# Set the HLT paths
import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOSiStripCalZeroBiasHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, # choose logical OR between Triggerbits
#    HLTPaths = [
#        #SiStripCalZeroBias
#        "HLT_ZeroBias",
#        #Random Trigger for Cosmic Runs
#        'RandomPath'
#        ],
    eventSetupPathsKey='SiStripCalZeroBias',
    throw = False # tolerate triggers stated above, but not available
)

# Select only events where tracker had HV on (according to DCS bit information)
# AND respective partition is in the run (according to FED information)
import CalibTracker.SiStripCommon.SiStripDCSFilter_cfi
DCSStatusForSiStripCalZeroBias = CalibTracker.SiStripCommon.SiStripDCSFilter_cfi.siStripDCSFilter.clone()

# Include masking only from Cabling and O2O
import CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi
siStripQualityESProducerUnbiased = CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi.siStripQualityESProducer.clone()
siStripQualityESProducerUnbiased.appendToDataLabel = 'unbiased'
siStripQualityESProducerUnbiased.ListOfRecordToMerge = cms.VPSet(
    cms.PSet(
        record = cms.string( 'SiStripDetCablingRcd' ), # bad components from cabling
        tag = cms.string( '' )
    ),
    cms.PSet(
        record = cms.string( 'SiStripBadChannelRcd' ), # bad components from O2O
        tag = cms.string( '' )
    )
)


# Clusterizer #
from RecoLocalTracker.SiStripClusterizer.SiStripClusterizer_cfi import *

siStripUnbiasedClusterizerConditions = SiStripClusterizerConditionsESProducer.clone(QualityLabel="unbiased", Label="unbiased")
calZeroBiasClusters = siStripClusters.clone()
if hasattr(calZeroBiasClusters, "Clusterizer"): calZeroBiasClusters.Clusterizer.ConditionsLabel = 'unbiased'

# Not persistent collections needed by the filters in the AlCaReco DQM
from DPGAnalysis.SiStripTools.eventwithhistoryproducerfroml1abc_cfi import *
from DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1tsDB_cfi import *

# SiStripQuality (only to test the different data labels)#
from CalibTracker.SiStripQuality.siStripQualityStatistics_cfi import siStripQualityStatistics
qualityStatistics = siStripQualityStatistics.clone(StripQualityLabel=cms.string("unbiased"))

# Sequence #
seqALCARECOSiStripCalZeroBias = cms.Sequence(ALCARECOSiStripCalZeroBiasHLT*DCSStatusForSiStripCalZeroBias*calZeroBiasClusters*APVPhases*consecutiveHEs)

## customizations for the pp_on_AA eras
from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
(pp_on_XeXe_2017 | pp_on_AA).toModify(ALCARECOSiStripCalZeroBiasHLT,
                                      eventSetupPathsKey='SiStripCalZeroBiasHI'
)

# Select pp-like events based on the pixel cluster multiplicity
import HLTrigger.special.hltPixelActivityFilter_cfi
HLTPixelActivityFilterForSiStripCalZeroBias = HLTrigger.special.hltPixelActivityFilter_cfi.hltPixelActivityFilter.clone()
HLTPixelActivityFilterForSiStripCalZeroBias.maxClusters = 500
HLTPixelActivityFilterForSiStripCalZeroBias.inputTag    = 'siPixelClusters'

seqALCARECOSiStripCalZeroBiasHI = cms.Sequence(ALCARECOSiStripCalZeroBiasHLT*HLTPixelActivityFilterForSiStripCalZeroBias*DCSStatusForSiStripCalZeroBias*calZeroBiasClusters*APVPhases*consecutiveHEs)

#Specify we want to use our other sequence 
(pp_on_XeXe_2017 | pp_on_AA).toReplaceWith(seqALCARECOSiStripCalZeroBias,
                                           seqALCARECOSiStripCalZeroBiasHI
)
