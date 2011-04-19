import FWCore.ParameterSet.Config as cms

################################################
# Configure program AnalyseDisplacedJetTrigger #
################################################

# My modified version allowing parallel running.
analysis = cms.EDAnalyzer('AnalyseDisplacedJetTrigger',

# Define good quality offline jets suitable for use in exotica displaced jet search.
# The kinematic cuts on the jets should be tight enough that they are likely to pass the
# the kinematic requirements of the displaced jet trigger.
# The cut on the number of prompt tracks should correspond to that used by the trigger.                          

   nPromptTkMax = cms.double(2.5),
   ptMin = cms.double(80.),
   etaMax = cms.double(1.95)                          
)
