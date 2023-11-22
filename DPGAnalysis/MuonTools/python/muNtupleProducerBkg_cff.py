import FWCore.ParameterSet.Config as cms

from PhysicsTools.NanoAOD.common_cff import *

from DPGAnalysis.MuonTools.nano_mu_global_cff import *
from DPGAnalysis.MuonTools.nano_mu_digi_cff import *

muDPGNanoProducerBkg = cms.Sequence(lhcInfoTableProducer
                                   + lumiTableProducer
                                   + muDigiProducersBkg)

def muDPGNanoBkgCustomize(process) :

     if hasattr(process, "NANOAODoutput"):
          process.NANOAODoutput.outputCommands.append("keep nanoaodFlatTable_*Table*_*_*")
          process.NANOAODoutput.outputCommands.append("drop edmTriggerResults_*_*_*")
     
     return process