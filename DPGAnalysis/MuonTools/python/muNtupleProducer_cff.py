import FWCore.ParameterSet.Config as cms

from PhysicsTools.NanoAOD.common_cff import *

from DPGAnalysis.MuonTools.nano_mu_global_cff import *
from DPGAnalysis.MuonTools.nano_mu_digi_cff import *
from DPGAnalysis.MuonTools.nano_mu_local_reco_cff import *
from DPGAnalysis.MuonTools.nano_mu_reco_cff import *
from DPGAnalysis.MuonTools.nano_mu_l1t_cff import *

muDPGNanoProducer = cms.Sequence(globalTables
                                + muDigiTables
                                + muLocalRecoTables
                                + muRecoTables
                                + muL1TriggerTables
                               )

def muDPGNanoCustomize(process) :

     if hasattr(process, "dtrpcPointFlatTable") and \
        hasattr(process, "cscrpcPointFlatTable") and \
        hasattr(process, "RawToDigiTask"):
          process.load("RecoLocalMuon.RPCRecHit.rpcPointProducer_cff")
          process.rpcPointProducer.dt4DSegments =  'dt4DSegments'
          process.rpcPointProducer.cscSegments =  'cscSegments'
          process.rpcPointProducer.ExtrapolatedRegion = 0.6
          process.RawToDigiTask.add(process.rpcPointProducer)

     if hasattr(process, "muGEMMuonExtTable") or hasattr(process, "muCSCTnPFlatTable"):
          process.load("TrackingTools/TransientTrack/TransientTrackBuilder_cfi")
          process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi")
          process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi")
          process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi")

     for output in ["NANOEDMAODoutput", "NANOAODoutput", "NANOEDMAODSIMoutput", "NANOAODSIMoutput"]:
          if hasattr(process, output) and "keep edmTriggerResults_*_*_*" in getattr(process,output).outputCommands:
               getattr(process,output).outputCommands.remove("keep edmTriggerResults_*_*_*")

     return process
