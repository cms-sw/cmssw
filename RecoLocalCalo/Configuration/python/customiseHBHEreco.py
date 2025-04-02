# Set M0 as default HCAL reconstruction over full HBHE Digi range
# To be used in cmsDriver command: 
# --customise RecoLocalCalo/Configuration/customiseHBHEreco.hbheUseM0FullRangePhase1

import FWCore.ParameterSet.Config as cms
def hbheUseM0FullRangePhase1(process):
    if hasattr(process,'hbhereco'):
       process.hbhereco.tsFromDB = False
       process.hbhereco.recoParamsFromDB = False
       process.hbhereco.sipmQTSShift = -99
       process.hbhereco.sipmQNTStoSum = 99
       process.hbhereco.algorithm.useMahi = False
       process.hbhereco.algorithm.useM2 = False
       process.hbhereco.algorithm.useM3 = False
       process.hbhereco.algorithm.correctForPhaseContainment = False
       process.hbhereco.algorithm.firstSampleShift = -999
       process.hbhereco.algorithm.samplesToAdd = 10

    return(process)
