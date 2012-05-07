
# En-able HF Noise filters in GRun menu
if 'hltHfreco' in locals():
    hltHfreco.setNoiseFlags = cms.bool( True )

# CMSSW version specific customizations
import os
cmsswVersion = os.environ['CMSSW_VERSION']

