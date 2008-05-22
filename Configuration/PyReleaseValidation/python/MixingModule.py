#G.Benelli
#This fragment is used to add the SimpleMemoryCheck
#and Timing services output to the log of the simulation
#performance candles.
#It is meant to be used with the cmsDriver.py option
#--customise in the following fashion:
#E.g.
#./cmsDriver.py HZZLLLL -e 190 -n 50 --step=GEN --customise=Simulation.py >& HZZLLLL_190_GEN.log&
#or
#./cmsDriver.py MINBIAS -n 50 --step=GEN --customise=Simulation.py >& MINBIAS_GEN.log&

import FWCore.ParameterSet.Config as cms
def customise(process):
    #Adding SimpleMemoryCheck service:
    process.SimpleMemoryCheck=cms.Service("SimpleMemoryCheck",
                                          ignoreTotal=cms.untracked.int32(1),
                                          oncePerEventMode=cms.untracked.bool(True))
    #Adding Timing service:
    process.Timing=cms.Service("Timing")

    #Overwriting the fileNames to be used by the MixingModule
    #when invoking cmsDriver.py with the --PU option
    process.mix.input.fileNames = cms.untracked.vstring('file:../MinBias_TimeSize/MINBIAS__GEN,SIM.root')

    return(process)
