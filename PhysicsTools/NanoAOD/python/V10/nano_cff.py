from __future__ import print_function
import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.nano_cff import *


def nanoAOD_customizeV10(process):
    
    # PUT here any old recipe that belonged to the V10 
    # update the PhysicsTools/NanoAOD/python as needed

    process.nanoSequence  = cms.Sequence(process.nanoSequence)
    process.nanoSequenceMC  = cms.Sequence(process.nanoSequenceMC)

    return process
