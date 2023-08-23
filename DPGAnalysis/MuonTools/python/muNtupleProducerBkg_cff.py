import FWCore.ParameterSet.Config as cms

from PhysicsTools.NanoAOD.common_cff import *

from DPGAnalysis.MuonTools.nano_mu_digi_cff import *

muNtupleProducerBkg = cms.Sequence(muDigiProducersBkg)

def nanoAOD_customizeCommon(process) :

     if hasattr(process, "NANOAODoutput"):
          process.NANOAODoutput.outputCommands.append("keep nanoaodFlatTable_*Table*_*_*")
          process.NANOAODoutput.outputCommands.append("drop edmTriggerResults_*_*_*")
     
     return process