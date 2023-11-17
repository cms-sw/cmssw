import FWCore.ParameterSet.Config as cms

from PhysicsTools.NanoAOD.common_cff import *

from DPGAnalysis.MuonTools.nano_mu_global_cff import *
from DPGAnalysis.MuonTools.nano_mu_digi_cff import *
from DPGAnalysis.MuonTools.nano_mu_local_reco_cff import *
from DPGAnalysis.MuonTools.nano_mu_reco_cff import *
from DPGAnalysis.MuonTools.nano_mu_l1t_cff import *
from DPGAnalysis.MuonTools.nano_mu_l1t_cff import *

muDPGNanoProducer = cms.Sequence(lhcInfoTableProducer
                                + lumiTableProducer
                                + muDigiProducers 
                                + muLocalRecoProducers 
                                + muRecoProducers
                                + muL1TriggerProducers
                               )
                               
def muDPGNanoCustomize(process) :
     
     if hasattr(process, "dtrpcPointFlatTableProducer") and \
        hasattr(process, "cscrpcPointFlatTableProducer") and \
        hasattr(process, "RawToDigiTask"):
          process.load("RecoLocalMuon.RPCRecHit.rpcPointProducer_cff")
          process.rpcPointProducer.dt4DSegments =  'dt4DSegments'
          process.rpcPointProducer.cscSegments =  'cscSegments'
          process.rpcPointProducer.ExtrapolatedRegion = 0.6
          process.RawToDigiTask.add(process.rpcPointProducer)
          
     if hasattr(process, "muGEMMuonExtTableProducer") or hasattr(process, "muCSCTnPFlatTableProducer"):
          process.load("TrackingTools/TransientTrack/TransientTrackBuilder_cfi")
          process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi")
          process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi")
          process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi")

     if hasattr(process, "NANOAODoutput"):
          process.NANOAODoutput.outputCommands.append("keep nanoaodFlatTable_*Table*_*_*")
          process.NANOAODoutput.outputCommands.append("drop edmTriggerResults_*_*_*")

     return process
