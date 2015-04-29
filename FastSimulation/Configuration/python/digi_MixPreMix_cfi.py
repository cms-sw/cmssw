import FWCore.ParameterSet.Config as cms

from SimGeneral.MixingModule.digi_MixPreMix_cfi import *

# fastsim has no castor
del theDigitizersMixPreMix.castor

# removing pixel and strips 
# usually done in Fastimulation.Configuration.MixigModule_Full2Fast.prepareDigiMixing
# but doing it here prevents mixing tracks in process.mix,
# so that they can be mixed in process.mixData 
del theDigitizersMixPreMix.pixel
del theDigitizersMixPreMix.strip

# same thing vor validation digitizers
del theDigitizersMixPreMixValid.pixel
del theDigitizersMixPreMixValid.strip
del theDigitizersMixPreMixValid.castor
