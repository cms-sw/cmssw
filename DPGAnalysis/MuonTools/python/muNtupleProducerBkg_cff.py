import FWCore.ParameterSet.Config as cms

from PhysicsTools.NanoAOD.common_cff import *

from DPGAnalysis.MuonTools.nano_mu_global_cff import *
from DPGAnalysis.MuonTools.nano_mu_digi_cff import *
from DPGAnalysis.MuonTools.nano_mu_local_reco_cff import *
from DPGAnalysis.MuonTools.nano_mu_reco_cff import *

muDPGNanoProducerBkg = cms.Sequence(globalTables
                                   + muDigiTablesBkg
                                   + muLocalRecoTablesBkg)

def muDPGNanoBkgCustomize(process) :

     for output in ["NANOEDMAODoutput", "NANOAODoutput", "NANOEDMAODSIMoutput", "NANOAODSIMoutput"]:
          if hasattr(process, output) and "keep edmTriggerResults_*_*_*" in getattr(process,output).outputCommands:
               getattr(process,output).outputCommands.remove("keep edmTriggerResults_*_*_*")
     
     return process
